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
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)

    d = np.sqrt(((x2 - x1) ** 2).sum())
    return d


# def load_history(path):
#     path = Path(path)
#
#     if not path.exists():
#         return []
#
#     history = []
#     with h5py.File(path, 'r') as hf:
#         for h_line in hf.keys():
#             hist_item = np.asarray(hf[h_line]).tolist()
#
#             history.append(hist_item)
#
#     return history
#
#
# def update_history(path, history):
#     path = Path(path)
#
#     with h5py.File(path, 'w') as hf:
#         for _, hist_item in enumerate(history):
#             item = np.atleast_2d(hist_item)
#             if item.shape == (1, 0):
#                 item = item.reshape((0, 2))
#
#             dset = hf.create_dataset(
#                 str(_),
#                 shape=item.shape,
#                 dtype=int
#             )
#
#             dset[0:] = item


def make_corner_plot(
        post, *, name, out_folder
):
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
    remove_coords = np.empty(shape=(0, 2))
    if force_remove is not None:
        remove_coords = force_remove
    else:
        for s in stars:
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
        b_u=None
):
    out_path = Path(out_path)

    fig, ax = plt.subplots()
    if show_mask:
        ax.imshow(field, cmap='GnBu', origin='upper')
        ax.imshow(masked_field, alpha=1, cmap='Greys', origin='upper')
    else:
        ax.imshow(field, cmap='Greys', origin='upper')

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
            # if hyp_2 is not 'b' but 's', the two hypotesis are switched.

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
