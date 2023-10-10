import cpnest
import h5py
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

from corner import corner

from pyfieldsim.core.stars import new_star, Star


plt.switch_backend('agg')

nlive = 500
maxmcmc = 5000


def dist(x1, x2):
    """
    Function to determine the distance between two stars.
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)

    d = np.sqrt(((x2 - x1) ** 2).sum())
    return d


def make_corner_plot(
        post, *, name, out_folder
):
    """
    Function to create a corner plot from a posterior distribution and saves it.
    """
    columns = [
        post[par] for par in post.dtype.names
        if par not in ['logL', 'logPrior']
    ]

    labels = [
        f'${par}$' for par in post.dtype.names
        if par not in ['logL', 'logPrior']
    ]

    has_two_stars = 'f' in post.dtype.names
    if has_two_stars:
        columns[1] = columns[0] * columns[1]
        labels[1] = '$A_1$'

    for _, l in enumerate(labels):
        for s in ['mu', 'sigma']:
            if l.find(s) > 0:
                labels[_] = l.replace(s, '\\' + s)
                l = labels[_]

        if (sub_p := l.find('_')) > 0:
            sub = l[sub_p + 1:-1]
            labels[_] = l.replace(sub, '{' + sub + '}')
            l = labels[_]

        if l.find('x') > 0:
            labels[_] = l.replace('x', 'y')
        elif l.find('y') > 0:
            labels[_] = l.replace('y', 'x')

    samples = np.column_stack(columns)

    c = corner(
        samples,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        use_math_text=True,
        show_titles=True,
        title_fmt='.3e'
    )
    c = update_title_fmts(c, post)
    c.savefig(
        out_folder.joinpath(f'joint_posterior_{name}.pdf'),
        bbox_inches='tight'
    )
    plt.close(c)


def _find_R(
        *,
        A, sigma, b_u,
        is_flat
):
    """
    Function to find the radius within which all pixels around a newly found
    star are masked. Depends on an upper limit on background set by the used.

    If the field is flat (contains no background) this function returns the
    width of the PSF, otherwise the smaller between the calculated radius
    and twice the PSF width.

    See the report for further information.
    """
    if is_flat:
        return sigma

    R = sigma * np.sqrt(
        -2 * np.log(
            (2 * np.pi * sigma) / A * b_u
        )
    )
    R = min(2 * sigma, R)

    return R


def mask_field(
        field, *,
        stars,
        mask,
        shape,
        sigma, b_u,
        is_flat,
        force_remove=None,
):
    """
    Function that masks the field for the next iteration of star
    identification.

    Parameters
    ----------
    field: `numpy.array` or `numpy.ma.masked_array`
        Containing the field to mask.
    stars: iterable of `Star` objects.
        The stars to be masked in the field at the current iteration.
    mask: `numpy.array` of the same shape of 'field'.
        The previous mask to be applied to the field.
    shape: `tuple`
        The shape of the field you are working on.
    sigma: `number`
        The width of the PSF on the field.
    b_u: `number`
        The upper level of background for a given star below which the star
        is removed.
    is_flat: `bool`
        A flag to identify a flat field
    force_remove: `iterable` or 'None'. Default is 'None'
        If passed, this function removes the given pixels from the field.

    Returns
    -------
    field: `numpy.ma.masked_array`
        The masked field.
    mask: `numpy.ndarray`
        The mask for the field.
    """
    remove_coords = np.empty(shape=(0, 2))
    if force_remove is not None:
        remove_coords = force_remove
    else:
        for s in stars:
            # loops through the newly identified stars to remove them. First
            # it finds the radius of pixels to be removed.
            R = _find_R(A=s.A, sigma=sigma, b_u=b_u, is_flat=is_flat)

            _remove_coords = [
                [x, y] for x in range(shape[0]) for y in range(shape[1])
                if dist([x, y], [s.mu[1], s.mu[0]]) <= R
                and (not np.ma.is_masked(field[x, y]) or field[x, y] > 0)
            ]
            _remove_coords = np.atleast_2d(_remove_coords)

            if _remove_coords.shape == (1, 0):
                _remove_coords = _remove_coords.reshape((0, 2))

            remove_coords = np.vstack((remove_coords, _remove_coords))

    # The passed mask is updated with the newly removed pixels.
    remove_coords = remove_coords.astype(int)
    for c in remove_coords:
        mask[c[0], c[1]] = True
    field = np.ma.array(
        data=field,
        mask=mask,
        fill_value=np.nan
    )

    return field, mask


def median_quantiles(qty, cl=1):
    """
    Function to determine the median of a quantity from its posterior
    distribution and the lower and upper confidence level depending on a
    'sigma-like' value of 1, 2 or 3.

    Parameters
    ----------
    qty: `iterable`
        Containing the posterior extractions of the quantity we want to find
        the median of.
    cl: '1', '2' or '3'. Default is '1'
        The confidence interval to return. 'cl=1' corresponds to the 16-84%
        interval, 'cl=2' to the '5-95%' interval and 'cl=3' to the '1-99%'
        interval.

    Returns
    -------
    q_50
        The median of the sample.
    err_m, err_p
        The errorbars for the lower and upper interval, respectively.
    fmt
        The format function to write the errorbars with the correct number
        of significant digits.

    Raises
    ------
    ValueError
        If 'cl' is different from '1', '2' or '3'.
    """
    if cl == 1:
        qtls = [0.16, 0.84]
    elif cl == 2:
        qtls = [0.05, 0.95]
    elif cl == 3:
        qtls = [0.01, 0.99]
    else:
        raise ValueError("Unexpected value.")

    q_50, q_l, q_u = np.quantile(qty, [0.5] + qtls)
    err_m = q_50 - q_l
    err_p = q_u - q_50

    min_error = min(err_m, err_p)
    odg_err = np.floor(np.log10(min_error)).astype(int)
    odg_meas = np.floor(np.log10(q_50)).astype(int)

    odg = f".{odg_meas - odg_err}e"
    fmt = f"{{0:{odg}}}".format

    return q_50, err_m, err_p, fmt


def plot_recovered_stars(
        field, *,
        stars,
        pos_errors,
        shape,
        out_path,
        show_sources,
        is_flat,
        sources=None,
        brt_coords=None,
        radius=None,
        show_mask=False,
        masked_field=None,
        sigma=None,
        b_u=None,
        forced=False
):
    """
    Function to plot the recovered stars over the field.

    Parameters
    ----------
    field: `numpy.ndarray`
        The field over which the recovered stars are plotted.
    stars: `iterable` of `Star` objects
        The list of recovered stars
    pos_errors: `iterable`
        Iterable of shape (N, 4), where N is the shape of parameter 'stars'.
    shape: `tuple`
        The shape of the field.
    out_path: `string` or `Path`-like object
        The path to the folder in which the plot will be saved.
    show_sources: `boolean`
        If 'True', the real sources are shown as green squares the size of a
        pixel. If 'False', the real sources are not shown.
    is_flat: `boolean`
        Wether the field contains background or not.
    sources: `iterable`
        An iterable of real sources in the field. It is used only is
        'show_sources=True'. Default is 'None'.
    brt_coords: `iterable` of length 2
        The coordinates of the selected bright pixel. Default is 'None'.
    radius: `number`
        The radius of pixels around the bright pixels that are eliminated in
        the iteration. Default is 'None'.
    show_mask: `boolean`
        Wether to show the masked pixels or not. Default is 'False'.
    masked_field: `numpy.ma.masked_array` or 'None'
        The field with the masked pixels from previous iterations. Default
        is None.
    sigma: `number`
        If 'show_mask=True', the sigma for the PSF to determine the mask
        radius. Default is 'False'.
    b_u: `number`
        The upper confidence level for the background to select the radius
        of the mask. Default is 'None'.
    forced: `boolean`
        If True, the pixels within `radius` from `brt_coords` are removed
        from the mask.
    """
    out_path = Path(out_path)

    fig, ax = plt.subplots()
    if show_mask:
        ax.imshow(field, cmap='GnBu', origin='upper', interpolation='none')
        img = ax.imshow(masked_field, alpha=1, cmap='Greys', origin='upper',
                        interpolation='none')
    else:
        img = ax.imshow(field, cmap='Greys', origin='upper',
                        interpolation='none')

    plt.colorbar(img)
    ax.set_aspect(1)

    if show_sources:
        for _, s in enumerate(sources):
            good_px = plt.Rectangle(
                (s[1] - 0.5, s[0] - 0.5),
                width=1, height=1,
                edgecolor='green',
                facecolor='none'
            )
            ax.add_artist(good_px)

            if _ == 0 and radius is not None:
                if forced:
                    valid, counts = select_valid_pixels(
                        field,
                        radius=radius,
                        shape=shape,
                        brt_coords=(stars[-1].mu[1], stars[-1].mu[0])
                    )
                    brt_index = np.argmax(counts)
                    brt_coords = valid[brt_index]

                circle = plt.Circle(
                    (brt_coords[1], brt_coords[0]),
                    radius=radius,
                    fill=False,
                    color='blue',
                    linewidth=0.5,
                    alpha=0.5,
                    linestyle='dashed'
                )
                ax.add_artist(circle)

    for s, err in zip(stars, pos_errors):
        x_errs = [[_] for _ in err[0]]
        y_errs = [[_] for _ in err[1]]

        if isinstance(s, Star) or show_mask:
            c = [s.mu[0], s.mu[1]]
        else:
            c = [s[1], s[0]]

        ax.errorbar(c[0], c[1],
                    xerr=x_errs,
                    yerr=y_errs,
                    fmt='.',
                    markersize=0.5,
                    color='red', elinewidth=0.7)

        if show_mask:
            R = _find_R(A=s.A, sigma=sigma, b_u=b_u, is_flat=is_flat)

            circle = plt.Circle(
                s.mu,
                radius=R,
                fill=False,
                color='red',
                linewidth=0.5,
                alpha=0.5,
                linestyle='solid'
            )
            ax.add_artist(circle)

    ax.set_xlim(-0.5, shape[1] - 0.5)
    ax.set_ylim(-0.5, shape[0] - 0.5)

    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def run_mcmc(
        model, *, name, out_folder='./', verbose=2, force=False
):
    """
    Function to run the MonteCarlo sampling for the inference.

    Parameters
    ----------
    model: `cpnest.model.Model`
        One of the models available in pyfieldsim.core.bayes
    name: `string`
        The name of the mcmc run to be used to save the samples.
    out_folder: `string` or `Path`-like object
        The output folder where to save the corner plot. Default is './'.
    verbose: int
        Verbosity level of the run. See cpnest.CPNest documentation. Default is '2'-
    force: `boolean`
        If 'True', the run is repeated overwriting a previous run. If
        `False`, the sampling is performed only if no folders with the same
        name are available. Default is `False`.

    Returns
    -------
    post: `numpy.ndarray`
        A structured array containing the posterior samples. See the
        documentation from cpnest.CPNest and related methods.
    logZ: `float`
        The logarithm of the evidence. To be used for model comparison.
    path: `Path`
        The output path of the folder for the samples.
    """
    path = Path(f'./samplings/sampling_output_{name}/')
    output = Path(path)
    out_folder = Path(out_folder)

    if (not output.exists()) or force:
        print("-- Inference run")
        work = cpnest.CPNest(
            model,
            verbose=verbose,
            nlive=nlive,  # 1000
            maxmcmc=maxmcmc,  # 5000
            nensemble=4,
            output=output,
        )
        work.run()
        post = work.posterior_samples.ravel()

        logZ = work.logZ
    else:
        print("-- Loading existing data")
        with h5py.File(output.joinpath('cpnest.h5'), 'r') as f:
            post = np.asarray(f['combined']['posterior_samples'])
            logZ = np.asarray(f['combined']['logZ']).reshape((1,))[0]

    make_corner_plot(post, name=name, out_folder=out_folder)

    return post, logZ, path


# noinspection PyArgumentList
def _select_1_2(
        post_1, post_2, *,
        stars, pos_errors,
        is_flat,
        select_1=True,
        b_cl=3,
        sigma=None
):
    # if flat field, this value is returned
    b_u = None

    if select_1:
        print("-- 1 star psf selected")
        sigma = np.median(post_1['sigma'])

        A_m, A_l, A_u, A_fmt = median_quantiles(post_1['A'])
        mu_y_m, mu_y_l, mu_y_u, y_fmt = median_quantiles(post_1['mu_y'])
        mu_x_m, mu_x_l, mu_x_u, x_fmt = median_quantiles(post_1['mu_x'])
        print(f"--- Star at [{y_fmt(mu_y_m)} (-{mu_y_l:.1e}, +{mu_y_u:.1e}),"
              f" {x_fmt(mu_x_m)} (-{mu_x_l:.1e}, +{mu_x_u:.1e})]")
        print(f"    of brightness {A_fmt(A_m)} (-{A_l:.1e} +{A_u:.1e})")

        stars.append(
            new_star(
                A=np.median(post_1['A']),
                mu=[
                    np.median(post_1['mu_y']),
                    np.median(post_1['mu_x'])
                ],
                sigma=sigma * np.eye(2),
                fmts=[A_fmt, y_fmt, x_fmt],
                pos_error=[mu_y_l, mu_y_u, mu_x_l, mu_x_u],
                A_error=[A_l, A_u]
            )
        )

        pos_errors.append([
            [mu_y_l, mu_y_u], [mu_x_l, mu_x_u]
        ])

        # for later use
        if not is_flat:
            b_m, b_l, b_u, b_fmt = median_quantiles(post_1['b'], cl=b_cl)
    else:
        print("-- 2 stars psf selected")
        sigma = np.median(post_2['sigma'])

        for _ in range(2):
            if _ == 0:
                A = np.median(post_2['A0'])
            else:
                A = np.median(post_2['A0'] * post_2['f'])

            if _ == 0:
                A_m, A_l, A_u, A_fmt = median_quantiles(post_2[f'A0'])
            else:
                A_m, A_l, A_u, A_fmt = median_quantiles(
                    post_2['f'] * post_2['A0']
                )

            mu_y_m, mu_y_l, mu_y_u, y_fmt = median_quantiles(
                post_2[f'mu_y{_}'])
            mu_x_m, mu_x_l, mu_x_u, x_fmt = median_quantiles(
                post_2[f'mu_x{_}'])
            print(
                f"--- Star at [{y_fmt(mu_y_m)} (-{mu_y_l:.1e} +{mu_y_u:.1e}),"
                f" {x_fmt(mu_x_m)} (-{mu_x_l:.1e} +{mu_x_u:.1e})]")
            print(
                f"    of brightness {A_fmt(A_m)} (-{A_l:.1e} +{A_u:.1e})")

            if _ > 0:
                f_m, f_l, f_u, f_fmt = median_quantiles(post_2['f'])
                print(
                    f"--- fraction is {f_fmt(f_m)} (-{f_l:.1e} +{f_u:.1e})"
                )

            stars.append(
                new_star(
                    A=A,
                    mu=[
                        np.median(post_2[f'mu_y{_}']),
                        np.median(post_2[f'mu_x{_}'])
                    ],
                    sigma=sigma * np.eye(2),
                    fmts=[A_fmt, y_fmt, x_fmt],
                    pos_error=[mu_y_l, mu_y_u, mu_x_l, mu_x_u],
                    A_error=[A_l, A_u]
                )
            )

            pos_errors.append([
                [mu_y_l, mu_y_u], [mu_x_l, mu_x_u]
            ])

        # for later use
        if not is_flat:
            b_m, b_l, b_u, b_fmt = median_quantiles(post_2['b'], cl=b_cl)

    return sigma, b_u


# noinspection PyArgumentList
def _select_s_b(
        post_1, post_2, *,
        stars, pos_errors,
        sigma,
        is_flat,
        select_1=True,
        b_cl=3
):
    # if flat field, this value is returned
    b_u = np.nan

    if select_1:  # or is_flat:
        print("-- Star identified")

        A_m, A_l, A_u, A_fmt = median_quantiles(post_1['A'])
        mu_y_m, mu_y_l, mu_y_u, y_fmt = median_quantiles(post_1['mu_y'])
        mu_x_m, mu_x_l, mu_x_u, x_fmt = median_quantiles(post_1['mu_x'])
        print(f"--- Star at [{y_fmt(mu_y_m)} (-{mu_y_l:.1e}, +{mu_y_u:.1e}),"
              f" {x_fmt(mu_x_m)} (-{mu_x_l:.1e}, +{mu_x_u:.1e})]")
        print(f"    of brightness {A_fmt(A_m)} (-{A_l:.1e} +{A_u:.1e})")

        stars.append(
            new_star(
                A=np.median(post_1['A']),
                mu=[
                    np.median(post_1['mu_y']),
                    np.median(post_1['mu_x'])
                ],
                sigma=sigma * np.eye(2),
                fmts=[A_fmt, y_fmt, x_fmt],
                pos_error=[mu_y_l, mu_y_u, mu_x_l, mu_x_u],
                A_error=[A_l, A_u]
            )
        )

        pos_errors.append([
            [mu_y_l, mu_y_u], [mu_x_l, mu_x_u]
        ])

        # for later use
        if not is_flat:
            b_m, b_l, b_u, b_fmt = median_quantiles(post_1['b'], cl=b_cl)
    else:
        print("-- Found background")
        b_u = np.nan

    return sigma, b_u


def select_hypothesis(
        *,
        hyp_1, hyp_2, logZ, logB_lim_hyp2,
        stars, pos_errors,
        post_1, post_2,
        is_flat,
        sigma=None,
):
    """
    Function that, given two hypotheses and their evidences, chooses between
    one of the two.

    Parameters
    ----------
    hyp_1: `string`
        The name of the first hypothesis.
    hyp_2: `string`
        The name of the second hypothesos
    logZ: `dict`
        A dictionary with keys given by `hyp_1` and `hyp_2` and values given by
        their evidences.
    logB_lim_hyp2: `float`
        The limit to the bayes factor over which to accept the hypothesis 1
        over the hypothesis 2.
    stars: `iterable` of `Star` objects
        The array of identified stars, where to store a newly identified star.
    pos_errors: `iterable` of shape (N, 1, 2)
        An iterable containing the errors on x and y positions, to add the
        one from the newly added star.
    post_1: `numpy.ndarray`
        The posterior samples from the first hypothesis.
    post_2: `numpy.ndarray`
        The posterior samples from the second hypothesis.
    is_flat: `boolean`
        Wether the field is flat or not.
    sigma: `float`
        The width of the PSF. Default is 'None'.

    Returns
    -------
    sigma: `float`
        The width of the PSF. If the two hypothesis are the single or double
        PSF star, `sigma` is inferred.
    b_u: `float`
        The upper background level limit at the 99% confidence level.
    hyp: `string`
        Either `hyp_1` or `hyp_2`, depending on which one is selected.
    """
    logBayesFactor = logZ[hyp_1] - logZ[hyp_2]

    print(f"--- logZ_{hyp_1} = {logZ[hyp_1]:.1f}")
    print(f"--- logZ_{hyp_2} = {logZ[hyp_2]:.1f}")
    print(f"--- logB = {logBayesFactor:.2f}")

    if 'b' in [hyp_1, hyp_2]:
        make_selection = _select_s_b

        if sigma is None:
            raise ValueError(
                "sigma cannot be `None`."
            )

        if not (hyp_2 == 'b'):
            # if hyp_2 is not 'b' but 's', the two hypothesis are switched.

            post_tmp = post_1.copy()
            post_1 = post_2.copy()
            post_2 = post_tmp
    else:
        make_selection = _select_1_2

    if logBayesFactor >= logB_lim_hyp2:
        return *make_selection(
            post_1, post_2,
            stars=stars, pos_errors=pos_errors,
            sigma=sigma,
            select_1=True,
            is_flat=is_flat
        ), hyp_1
    else:
        return *make_selection(
            post_1, post_2,
            stars=stars, pos_errors=pos_errors,
            sigma=sigma,
            select_1=False,
            is_flat=is_flat
        ), hyp_2


def select_valid_pixels(
        field, *, radius, shape, brt_coords
):
    """
    Function to select the coordinates for the valid pixels from a field,
    around a bright point and within a certain radius.

    Parameters
    ----------
    field: `numpy.ndarray`
        The imaged field.
    radius: `float`
        The radius within which pixels are selected.
    shape: `tuple`
        The shape of the field.
    brt_coords: iterable of length 2
        The coordinates for the point around which pixels should be selected.

    Returns
    -------
    valid_coords: `numpy.ndarray`
        The array of coordinates of selected pixels within `radius` distance
        from `brt_coords`.
    valid_counts: `numpy.ndarray`
        The array of counts at those coordinates.

    """
    valid_coords = np.array([
        [x, y] for x in range(shape[0]) for y in range(shape[1])
        if dist([x, y], brt_coords) <= radius and field[x, y] > 0
    ])  # see appendix C

    valid_counts = np.array([
        field[x[0], x[1]] for x in valid_coords
    ])

    return valid_coords, valid_counts


# noinspection PyArgumentList,PyProtectedMember
def update_title_fmts(c, post):
    """
    Function to update the format functions for the strings containing the
    errorbars of data.
    """
    dim_space = np.sqrt(len(c.axes)).astype(int)
    names = post.dtype.names

    n_ = 0
    for _, ax in enumerate(c.axes):
        if (_ == 0) or (_ % (dim_space+1) == 0):
            name = names[n_]

            if name == 'f':
                val_50, val_m, val_p, fmt = median_quantiles(
                    post['A0'] * post['f']
                )
            else:
                val_50, val_m, val_p, fmt = median_quantiles(post[name])

            old_title = ax.title._text

            n_idx = old_title.find("$")
            n_idx_end = old_title.find("$", n_idx + 1, -1)

            name_title = rf"${old_title[n_idx + 1:n_idx_end]}$ = "
            val_title = r"${{{0}}}".format(fmt(val_50))
            err_m_title = r"_{{{0}}}".format(f"-{val_m:.0e}")
            err_p_title = r"^{{{0}}}$".format(f"+{val_p:.0e}")

            title = name_title + val_title + err_m_title + err_p_title

            ax.set_title(title)
            n_ += 1

    return c
