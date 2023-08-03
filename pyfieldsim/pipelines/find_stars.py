import argparse as ag
import sys

import cpnest
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

from corner import corner
from pathlib import Path

from pyfieldsim.core.fieldtypes.field import Field
from pyfieldsim.core.stars import new_star

from pyfieldsim.core.bayes import FindPsf, FindPsf2
from pyfieldsim.core.bayes import FindStar, FindBackground

from pyfieldsim.utils.metadata import read_metadata
from pyfieldsim.utils.save_stars import save_stars


def main():
    parser = ag.ArgumentParser(
        prog='fs-find-stars',
        description='',
    )
    parser.add_argument('data_file')
    parser.add_argument("-w", "--width", type=float,
                        dest='initial_width', required=True,
                        help="")
    parser.add_argument("-f", "--flat", action='store_true',
                        dest='is_flat', default=False,
                        help="")
    parser.add_argument("-s", "--sources", action='store_true',
                        dest='show_sources', default=False,
                        help="")
    parser.add_argument("-o", "--output",
                        dest='out_folder', default=None,
                        help="")
    parser.add_argument("--options",
                        dest='options', default=None,
                        help="")

    args = parser.parse_args()

    if args.out_folder is None:
        out_folder = Path(os.getcwd())
    else:
        out_folder = Path(args.out_folder)

    data_file = Path(args.data_file)

    sources_file = Path('S' + data_file.name[1:])
    sources_metadata = read_metadata(sources_file)

    shape = sources_metadata['shape']
    pad = sources_metadata['pad']

    with h5py.File(sources_file, 'r') as f:
        coords = np.asarray(f['coords'])

    for s in coords:
        s[0] -= pad[0]
        s[1] -= pad[1]

    obs_stars = []
    for idx, s in enumerate(coords):
        if 0 <= s[0] < shape[0] and 0 <= s[1] < shape[1]:
            obs_stars.append(idx)

    obs_stars = np.asarray(obs_stars)
    coords = coords[obs_stars]

    if not args.is_flat:
        bkgnd_analysis_file = data_file.with_suffix('')
        bkgnd_analysis_file = Path(bkgnd_analysis_file.name
                                   + '_bkg_analysis.h5')
        bkgnd_analysis_metadata = read_metadata(bkgnd_analysis_file)
        limit_points = 3
    else:
        bkgnd_analysis_metadata = {
            'mean': 0,
            'std': 1
        }
        limit_points = 4

    data_field = Field.from_field(data_file)

    # 1 - Find the brightest star. Take all points in a circle around some
    # user-defined distance from the bright point.
    brt = data_field.field.max()
    brt_coords = np.unravel_index(np.argmax(data_field.field), shape)

    initial_radius = min(args.initial_width, 10)

    valid_coords = np.array([
        [x, y] for x in range(shape[0]) for y in range(shape[1])
        if dist([x, y], brt_coords) <= initial_radius
        and data_field.field[x, y] > 0
    ])  # see appendix D
    valid_counts = np.array([
        data_field.field[x[0], x[1]] for x in valid_coords
    ])

    # 2 - Set bounds for mean at +- 1 px from the brightest point. Set
    # bounds for sigma. NB: should be prior, can't watch the data!
    mins = np.min(valid_coords, axis=0)
    maxs = np.max(valid_coords, axis=0)
    m_bounds = [
        [mins[0], maxs[0]],
        [mins[1], maxs[1]]
    ]
    s_bounds = [
        [0, 10]
    ]

    # 3 - Set bounds for brightness. 0 to 100 * max_value. A is surely
    # inside.
    A_bounds = [
        [brt / 10, 100 * brt]
    ]
    b_bounds = [[
        max(
            bkgnd_analysis_metadata['mean']
                - 2 * bkgnd_analysis_metadata['std'], 1),
        bkgnd_analysis_metadata['mean'] + bkgnd_analysis_metadata['std']
    ]]

    # 4 - Find one star within this radius.
    bounds = A_bounds + m_bounds + s_bounds + b_bounds

    fit_model_1 = FindPsf(valid_coords, valid_counts,
                          bkgnd_analysis_metadata,
                          bounds,
                          is_flat=args.is_flat)

    psf1_path = Path('./sampling_output_psf1/')
    print("- Testing 1 star psf")
    if not psf1_path.exists():
        print("-- Inference run")
        work = cpnest.CPNest(
            fit_model_1,
            verbose=2,
            nlive=1000,  # 1000
            maxmcmc=5000,  # 5000
            nensemble=4,
            output=psf1_path,
        )
        work.run()
        post_1 = work.posterior_samples.ravel()

        logZ_1 = work.logZ
    else:
        print("-- Loading existing data")
        with h5py.File(psf1_path.joinpath('cpnest.h5'), 'r') as f:
            post_1 = np.asarray(f['combined']['posterior_samples'])

            logZ_1 = np.asarray(f['combined']['logZ']).reshape((1,))[0]

    logZ_psf = {
        '1': logZ_1
    }

    columns_1 = [
        post_1[par] for par in post_1.dtype.names
        if par not in ['logL', 'logPrior']
    ]
    samples_1 = np.column_stack(columns_1)
    labels = [
        f'${par}$' for par in post_1.dtype.names
        if par not in ['logL', 'logPrior']
    ]
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

    c = corner(
        samples_1,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        use_math_text=True,
        show_titles=True,
        title_fmt='.3e'
    )
    c = update_title_fmts(c, post_1)
    c.savefig(
        out_folder.joinpath(f'joint_posterior_psf1.pdf'),
        bbox_inches='tight'
    )

    # 4b - Check against two or three gaussians in the same dataset. Compare
    # evidences.
    bounds = A_bounds + m_bounds + s_bounds + b_bounds

    fit_model_2 = FindPsf2(valid_coords, valid_counts,
                           bkgnd_analysis_metadata, bounds,
                           is_flat=args.is_flat)

    psf2_path = Path('./sampling_output_psf2/')
    print("- Testing 2 stars psf")
    if initial_radius < 2:
        logZ_2 = -np.inf
    elif not psf2_path.exists():
        print("-- Inference run")
        work = cpnest.CPNest(
            fit_model_2,
            verbose=2,
            nlive=1000,  # 1000
            maxmcmc=5000,  # 5000
            nensemble=1,
            output='./sampling_output_psf2/',
        )
        work.run()
        post_2 = work.posterior_samples.ravel()

        logZ_2 = work.logZ
    else:
        print("-- Loading existing data")
        with h5py.File(psf2_path.joinpath('cpnest.h5'), 'r') as f:
            post_2 = np.asarray(f['combined']['posterior_samples'])
            logZ_2 = np.asarray(f['combined']['logZ']).reshape((1,))[0]

    logZ_psf['2'] = logZ_2

    try:
        s0_s1_dist = dist(
            (coords[0][1], coords[0][0]),
            (coords[1][1], coords[1][0])
        )
    except IndexError:
        pass
    else:
        if s0_s1_dist > initial_radius:
            logZ_psf['2'] = -np.inf

    if np.isfinite(logZ_psf['2']):
        columns_2 = np.asarray([
            post_2[par] for par in post_2.dtype.names
            if par not in ['logL', 'logPrior']
        ])
        columns_2[1] = columns_2[0] * columns_2[1]
        labels = [f'${par}$' for par in post_2.dtype.names
                  if par not in ['logL', 'logPrior']]
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

        samples_2 = np.column_stack(columns_2)

        c = corner(
            samples_2,
            labels=labels,
            quantiles=[0.16, 0.5, 0.84],
            use_math_text=True,
            show_titles=True,
            title_fmt='.3e'
        )
        c = update_title_fmts(c, post_2)
        c.savefig(
            out_folder.joinpath(f'joint_posterior_psf2.pdf'),
            bbox_inches='tight'
        )

    stars = []
    pos_errors = []
    logBayesFactor = logZ_psf['1'] - logZ_psf['2']
    # 10^0.5 -> substantial evidence
    # 10^1 -> strong evidence
    # 10^1.5 -> very strong evidence
    # 10^2 -> decisive evidence

    print(f"--- logZ_1star = {logZ_psf['1']:.1f}")
    print(f"--- logZ_2star = {logZ_psf['2']:.1f}")
    print(f"--- logB = {logBayesFactor:.2f}")

    if logBayesFactor >= -1 * np.log(10):
        print("-- 1 star psf selected")
        sigma = np.median(post_1['sigma'])
        stars.append(
            new_star(
                A=np.median(post_1['A']),
                mu=[
                    np.median(post_1['mu_y']),
                    np.median(post_1['mu_x'])
                ],
                sigma=sigma * np.eye(2)
            )
        )
        A_m, A_l, A_u, A_fmt = median_quantiles(post_1['A'])
        mu_y_m, mu_y_l, mu_y_u, y_fmt = median_quantiles(post_1['mu_y'])
        mu_x_m, mu_x_l, mu_x_u, x_fmt = median_quantiles(post_1['mu_x'])
        print(f"--- Star at [{y_fmt(mu_y_m)} (-{mu_y_l:.1e}, +{mu_y_u:.1e}),"
              f" {x_fmt(mu_x_m)} (-{mu_x_l:.1e}, +{mu_x_u:.1e})]")
        print(f"    of brightness {A_fmt(A_m)} (-{A_l:.1e} +{A_u:.1e})")

        pos_errors.append([
            [mu_y_l, mu_y_u], [mu_x_l, mu_x_u]
        ])

        # for later use
        b_m, b_l, b_u, b_fmt = median_quantiles(post_1['b'], cl=3)
    else:
        print("-- 2 stars psf selected")
        sigma = np.median(post_2['sigma'])
        for _ in range(2):
            if _ == 0:
                A = np.median(post_2['A0'])
            else:
                A = np.median(post_2['A0'] * post_2['f'])

            stars.append(
                new_star(
                    A=A,
                    mu=[
                        np.median(post_2[f'mu_y{_}']),
                        np.median(post_2[f'mu_x{_}'])
                    ],
                    sigma=sigma * np.eye(2)
                )
            )
        A0_m, A0_l, A0_u, A0_fmt = median_quantiles(post_2['A0'])
        mu_y0_m, mu_y0_l, mu_y0_u, y0_fmt = median_quantiles(post_2['mu_y0'])
        mu_x0_m, mu_x0_l, mu_x0_u, x0_fmt = median_quantiles(post_2['mu_x0'])
        print(
            f"--- Star at [{y0_fmt(mu_y0_m)} (-{mu_y0_l:.1e} +{mu_y0_u:.1e}),"
            f" {x0_fmt(mu_x0_m)} (-{mu_x0_l:.1e} +{mu_x0_u:.1e})]")
        print(f"    of brightness {A0_fmt(A0_m)} (-{A0_l:.1e} +{A0_u:.1e})")

        A1_m, A1_l, A1_u, A1_fmt = median_quantiles(post_2['f'] * post_2['A0'])
        mu_y1_m, mu_y1_l, mu_y1_u, y1_fmt = median_quantiles(post_2['mu_y1'])
        mu_x1_m, mu_x1_l, mu_x1_u, x1_fmt = median_quantiles(post_2['mu_x1'])
        print(
            f"--- Star at [{y1_fmt(mu_y1_m)} (-{mu_y1_l:.1e} +{mu_y1_u:.1e}),"
            f" {x1_fmt(mu_x1_m)} (-{mu_x1_l:.1e} +{mu_x1_u:.1e})]")
        print(f"    of brightness {A1_fmt(A1_m)} (-{A1_l:.1e} +{A1_u:.1e})")

        f_m, f_l, f_u, f_fmt = median_quantiles(post_2['f'])
        print(
            f"--- fraction is {f_fmt(f_m)} (-{f_l:.1e} +{f_u:.1e})"
        )

        pos_errors.append([
            [mu_y0_l, mu_y0_u], [mu_x0_l, mu_x0_u]
        ])
        pos_errors.append([
            [mu_y1_l, mu_y1_u], [mu_x1_l, mu_x1_u]
        ])

        # for later use
        b_m, b_l, b_u, b_fmt = median_quantiles(post_2['b'], cl=3)

    pos_errors = np.array(pos_errors)

    fig, ax = plt.subplots()
    ax.imshow(data_field.field, cmap='Greys', origin='upper')
    ax.set_aspect(1)

    if args.show_sources:
        for _, s in enumerate(coords):
            good_px = plt.Rectangle(
                (s[1] - 0.5, s[0] - 0.5),
                width=1, height=1,
                edgecolor='green',
                facecolor='none'
            )
            ax.add_artist(good_px)

            if _ == 0:
                circle = plt.Circle(
                    tuple(brt_coords),
                    radius=initial_radius,
                    fill=False,
                    color='blue',
                    linewidth=0.5,
                    alpha=0.5,
                    linestyle='dashed'
                )
                ax.add_artist(circle)

    for s, err in zip(stars, pos_errors):
        x_errs = err[0].reshape((2, 1))
        y_errs = err[1].reshape((2, 1))

        ax.errorbar(s.mu[0], s.mu[1],
                    xerr=x_errs,
                    yerr=y_errs,
                    fmt='.',
                    markersize=0.5,
                    color='red', elinewidth=0.7)

    ax.set_xlim(-0.5, shape[1] - 0.5)
    ax.set_ylim(-0.5, shape[0] - 0.5)

    fig.savefig(
        out_folder.joinpath(f'recovered_psf_star.pdf'),
        bbox_inches='tight'
    )

    # 5 - Remove all points within n * s from the mean (or each mean) from the
    # dataset.
    # TODO: set min(n * s, R) with R from background (see calcs). n should
    #  be the psf sigma limit after which algorithm cannot recognise two
    #  identical (or almost identical) stars.

    # Radius from which to remove points
    remove_coords = np.empty(shape=(0, 2))
    valid_coords_mask = np.zeros(shape).astype(bool)
    for s in stars:
        R = sigma * np.sqrt(
            -2 * np.log(
                (2 * np.pi * sigma) / s.A * b_u
            )
        )
        R = min(1 * sigma, R)

        _remove_coords = np.array([
            [x, y] for x in range(shape[0]) for y in range(shape[1])
            if dist([x, y], [s.mu[1], s.mu[0]]) <= R
            and data_field.field[x, y] > 0
        ])
        remove_coords = np.concatenate((remove_coords, _remove_coords))

    remove_coords = remove_coords.astype(int)
    for c in remove_coords:
        valid_coords_mask[c[0], c[1]] = True
    field = np.ma.array(
        data=data_field.field,
        mask=valid_coords_mask,
        fill_value=np.nan
    )
    #
    plt.imshow(field, alpha=0.5)
    for s in stars:
        R = sigma * np.sqrt(
            -2 * np.log(
                (2 * np.pi * sigma) / s.A * b_u
            )
        )
        R = min(1 * sigma, R)

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
    plt.show()

    # TODO: remove before flight
    sys.exit(0)

    # 6 - Find next brightest star and select all points in a circle around
    # 5σ from the bright point.
    reached_background = False
    star_id = 1

    while not reached_background:
        print(f"- Finding star #{star_id}")
        brt = field.max()
        if args.is_flat and brt == 0:
            print("-- Reached background")
            reached_background = True
            continue
        elif isinstance(brt, np.ma.core.MaskedConstant):
            print("-- Reached background")
            reached_background = True
            continue

        brt_coords = np.unravel_index(np.argmax(field), shape)

        # TODO: Check if 1.5 creates aliases.
        valid_coords = np.array([
            [x, y] for x in range(shape[0]) for y in range(shape[1])
            if dist([x, y], brt_coords) <= 1.5 * sigma and field[x, y] > 0
        ])  # check if 1.5 creates aliases
        valid_counts = np.array([
            field[x[0], x[1]] for x in valid_coords
        ])

        if valid_coords.shape[0] <= limit_points:
            for c in valid_coords:
                valid_coords_mask[c[0], c[1]] = True

            field = np.ma.array(
                data=field,
                mask=valid_coords_mask,
                fill_value=np.nan
            )
            print("-- Too few points to infer a star")
            continue

        # 7 - Check against background only, 1, 2 or 3 stars in the save
        # dataset. Compare evidences.
        m_bounds = [
            [brt_coords[0] - 1, brt_coords[0] + 1],
            [brt_coords[1] - 1, brt_coords[1] + 1]
        ]
        A_bounds = [
            [0, 100 * brt]
        ]
        b_bounds = [
            [0,
             bkgnd_analysis_metadata['mean']
                + 3 * bkgnd_analysis_metadata['std']]
        ]

        bounds = A_bounds + m_bounds + b_bounds

        fit_model_s = FindStar(
            valid_coords, valid_counts,
            bkgnd_analysis_metadata,
            bounds,
            sigma,
            is_flat=args.is_flat
        )

        star_path = Path('./sampling_output_star/')
        print("-- Testing against star hypothesis")
        work = cpnest.CPNest(
            fit_model_s,
            verbose=1,
            nlive=1000,  # 1000
            maxmcmc=5000,  # 5000
            nensemble=4,
            output=star_path,
        )
        work.run()
        post_s = work.posterior_samples.ravel()

        logZ = {
            's': work.logZ
        }

        columns_s = [post_s[par] for par in post_s.dtype.names
                     if par not in ['logL', 'logPrior']]
        samples_s = np.column_stack(columns_s)

        if not args.is_flat:
            print("-- Testing against pure background hypothesis")
            bounds = b_bounds
            fit_model_b = FindBackground(
                valid_coords, valid_counts,
                bkgnd_analysis_metadata,
                bounds
            )

            background_path = Path('./sampling_output_bgnd/')
            work = cpnest.CPNest(
                fit_model_b,
                verbose=0,
                nlive=100,  # 1000
                maxmcmc=5000,  # 5000
                nensemble=1,
                output=background_path,
            )
            work.run()
            post_b = work.posterior_samples.ravel()

            logZ['b'] = work.logZ

            columns_b = [post_2[par] for par in post_b.dtype.names
                         if par not in ['logL', 'logPrior']]
            samples_b = np.column_stack(columns_b)

        if (not args.is_flat and logZ['s'] >= logZ['b']) or args.is_flat:
            stars.append(
                new_star(
                    A=np.median(post_s['A']),
                    mu=[
                        np.median(post_s['mu_y']),
                        np.median(post_s['mu_x'])
                    ],
                    sigma=sigma * np.eye(2)
                )
            )
            print(f"--- Star at ({np.median(post_s['mu_y'])}, "
                  f"{np.median(post_s['mu_x'])}), "
                  f"of brightness {np.median(post_s['A'])}")
        else:
            print("-- Reached background")
            reached_background = True
            continue

        # 8 - Remove all points within 3s from the mean (or each mean) from the
        # dataset.
        new_star_coords = [np.median(post_s['mu_x']),
                           np.median(post_s['mu_y'])]
        remove_coords = np.array([
            [x, y] for x in range(shape[0]) for y in range(shape[1])
            if dist([x, y], new_star_coords) <= 3 * sigma
        ])
        for c in remove_coords:
            valid_coords_mask[c[0], c[1]] = True
        field = np.ma.array(
            data=field,
            mask=valid_coords_mask,
            fill_value=np.nan
        )

        fig, ax = plt.subplots()
        ax.imshow(data_field.field, cmap='Greys', origin='upper')
        if args.show_sources:
            for s in coords:
                ax.scatter(s[1], s[0],
                           marker='o', edgecolor='green', facecolor='none')
        for s in stars:
            ax.scatter(s.mu[0], s.mu[1], marker='+', color='red')
        ax.set_xlim(0, shape[1])
        ax.set_ylim(0, shape[0])

        fig.savefig(
            out_folder.joinpath(f'recovered_stars_star{star_id}.pdf'),
            bbox_inches='tight'
        )

        star_id += 1

        # 9 - Iterate from 6# until background term dominates in a dataset.

    # TODO: Correlate points within deletion limit and some more (twice?)
    #  from each star to check for aliases. Maybe use `correlate` from
    #  `scipy.signal` as correlate([[star_c_x, star_c_y]], [s1, s2, s3],
    #  ...), with s_i = [s_i_x, s_i_y], s_i being the stars inside that
    #  region.

    # 10 - Profit.
    fig, ax = plt.subplots()
    ax.imshow(data_field.field, cmap='Greys', origin='upper')
    if args.show_sources:
        for s in coords:
            ax.scatter(s[1], s[0],
                       marker='o', edgecolor='green', facecolor='none')
    for s in stars:
        ax.scatter(s.mu[0], s.mu[1], marker='+', color='red')
    ax.set_xlim(0, shape[1])
    ax.set_ylim(0, shape[0])

    fig.savefig(
        out_folder.joinpath(f'recovered_stars_all.pdf'),
        bbox_inches='tight'
    )

    save_stars(stars, data_file, options=args.options)


def dist(x1, x2):
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)

    d = np.sqrt(((x2 - x1) ** 2).sum())
    return d


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

    odg = f".{odg_meas - odg_err + 1}e"
    fmt = f"{{0:{odg}}}".format

    return q_50, err_m, err_p, fmt


def update_title_fmts(c, post):
    dim_space = np.sqrt(len(c.axes)).astype(int)
    names = post.dtype.names

    n_ = 0
    for _, ax in enumerate(c.axes):
        if (_ == 0) or (_ % (dim_space+1) == 0):
            name = names[n_]
            val_50, val_m, val_p, fmt = median_quantiles(post[name])

            old_title = ax.title._text

            n_idx = old_title.find("$")
            n_idx_end = old_title.find("$", n_idx + 1, -1)

            name_title = rf"${old_title[n_idx + 1:n_idx_end]}$ = "
            val_title = r"${{{0}}}".format(fmt(val_50))
            err_m_title = r"_{{{0}}}".format(f"{val_m:.1e}")
            err_p_title = r"^{{{0}}}$".format(f"{val_p:.1e}")

            title = name_title + val_title + err_m_title + err_p_title

            ax.set_title(title)
            n_ += 1

    return c


if __name__ == "__main__":
    main()
