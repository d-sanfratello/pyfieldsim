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

from pyfieldsim.core.stars.find_utils import (
    dist,
    mask_field,
    plot_recovered_stars,
    run_mcmc,
    select_hypothesis,
    select_valid_pixels,
)

from pyfieldsim.core.bayes import FindPsf, FindPsf2
from pyfieldsim.core.bayes import FindStar, FindBackground

from pyfieldsim.utils.metadata import read_metadata
from pyfieldsim.utils.save_stars import save_stars


# noinspection PyArgumentList
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
    plot_folder = out_folder.joinpath('plots')
    samplings_folder = out_folder.joinpath('samplings')

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

    valid_coords, valid_counts = select_valid_pixels(
        data_field.field,
        radius=initial_radius,
        shape=shape,
        brt_coords=brt_coords
    )

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

    print("- Testing 1 star psf")
    post_1, logZ_1 = run_mcmc(
        model=fit_model_1,
        name='psf1',
        out_folder=plot_folder,
        verbose=2,
        force=False
    )
    logZ_psf = {'1': logZ_1}

    # 4b - Check against two or three gaussians in the same dataset. Compare
    # evidences.
    bounds = A_bounds + m_bounds + s_bounds + b_bounds

    fit_model_2 = FindPsf2(valid_coords, valid_counts,
                           bkgnd_analysis_metadata, bounds,
                           is_flat=args.is_flat)

    print("- Testing 2 stars psf")
    if initial_radius < 2:
        logZ_2 = -np.inf
        post_2 = None
    else:
        post_2, logZ_2 = run_mcmc(
            model=fit_model_2,
            name='psf2',
            out_folder=plot_folder,
            verbose=2,
            force=False
        )

    # If the two stars lies farther than the initial radius from each other,
    # the two stars hypothesis is discarded.
    try:
        s0_s1_dist = dist(
            (coords[0][1], coords[0][0]),
            (coords[1][1], coords[1][0])
        )
    except IndexError:
        pass
    else:
        if s0_s1_dist > initial_radius:
            logZ_2 = -np.inf

    logZ_psf['2'] = logZ_2

    stars = []
    pos_errors = []

    # 10^0.5 -> substantial evidence
    # 10^1 -> strong evidence
    # 10^1.5 -> very strong evidence
    # 10^2 -> decisive evidence
    sigma, b_u = select_hypothesis(
        hyp_1='1', hyp_2='2',
        logZ=logZ_psf,
        logB_lim_hyp2=-1 * np.log(10),
        stars=stars,
        pos_errors=pos_errors,
        post_1=post_1,
        post_2=post_2
    )

    fig = plot_recovered_stars(
        data_field.field,
        stars=stars,
        pos_errors=pos_errors,
        shape=shape,
        show_sources=args.show_sources,
        sources=coords,
        brt_coords=brt_coords,
        radius=initial_radius,
        out_path=plot_folder.joinpath('recovered_psf_star.pdf')
    )

    # 5 - Remove all points within R from each mean from the dataset.
    valid_coords_mask = np.zeros(shape).astype(bool)
    field, valid_coords_mask = mask_field(
        data_field.field,
        stars=stars,
        mask=valid_coords_mask,
        shape=shape,
        sigma=sigma, b_u=b_u
    )

    fig = plot_recovered_stars(
        data_field.field,
        stars=stars,
        pos_errors=pos_errors,
        shape=shape,
        out_path=plot_folder.joinpath(f'removed_pixels_psf_star.pdf'),
        show_sources=True,
        sources=coords,
        brt_coords=brt_coords,
        radius=initial_radius,
        show_mask=True,
        masked_field=field,
        sigma=sigma,
        b_u=b_u
    )

    # TODO: remove before flight
    sys.exit(0)

    # 6 - Find next brightest star and select all points in a circle around
    # 5σ from the bright point.
    exit_loop = False
    star_id = len(stars) + 1

    while not exit_loop:
        print(f"- Finding star #{star_id}")
        brt = field.max()
        if args.is_flat and brt == 0:
            print("-- Reached end")
            exit_loop = True
            continue
        elif isinstance(brt, np.ma.core.MaskedConstant):
            print("-- Reached end")
            exit_loop = True
            continue

        brt_coords = np.unravel_index(np.argmax(field), shape)

        # TODO: Check if creates aliases.
        radius = min(3 * sigma, 10)

        valid_coords, valid_counts = select_valid_pixels(
            field, radius=radius,
            shape=shape,
            brt_coords=brt_coords
        )

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

        # 7 - Check against background only or 1 star in the dataset.
        # Compare evidences.
        m_bounds = [
            [brt_coords[0] - 1, brt_coords[0] + 1],
            [brt_coords[1] - 1, brt_coords[1] + 1]
        ]
        A_bounds = [
            [brt / 10, 100 * brt]
        ]
        b_bounds = [[
            max(bkgnd_analysis_metadata['mean']
                - 2 * bkgnd_analysis_metadata['std'], 1),
            bkgnd_analysis_metadata['mean'] + bkgnd_analysis_metadata['std']
        ]]

        bounds = A_bounds + m_bounds + b_bounds

        fit_model_s = FindStar(
            valid_coords, valid_counts,
            bkgnd_analysis_metadata,
            bounds,
            sigma,
            is_flat=args.is_flat
        )

        star_path = Path(f'./sampling_output_star_{star_id}/')
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

            background_path = Path(f'./sampling_output_bgnd_{star_id}/')
            work = cpnest.CPNest(
                fit_model_b,
                verbose=0,
                nlive=1000,  # 1000
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
            s = new_star(
                    A=np.median(post_s['A']),
                    mu=[
                        np.median(post_s['mu_y']),
                        np.median(post_s['mu_x'])
                    ],
                    sigma=sigma * np.eye(2)
                )
            stars.append(s)
            A_m, A_l, A_u, A_fmt = median_quantiles(post_s['A'])
            mu_y_m, mu_y_l, mu_y_u, y_fmt = median_quantiles(post_s['mu_y'])
            mu_x_m, mu_x_l, mu_x_u, x_fmt = median_quantiles(post_s['mu_x'])
            print(
                f"--- Star at [{y_fmt(mu_y_m)} (-{mu_y_l:.1e}, +{mu_y_u:.1e}),"
                f" {x_fmt(mu_x_m)} (-{mu_x_l:.1e}, +{mu_x_u:.1e})]")
            print(f"    of brightness {A_fmt(A_m)} (-{A_l:.1e} +{A_u:.1e})")

            pos_errors.append([
                [mu_y_l, mu_y_u], [mu_x_l, mu_x_u]
            ])

            # for later use
            b_m, b_l, b_u, b_fmt = median_quantiles(post_s['b'], cl=3)

            # 8 - Remove all points within R from the mean from the dataset.
            R = sigma * np.sqrt(
                -2 * np.log(
                    (2 * np.pi * sigma) / s.A * b_u
                )
            )
            R = min(1 * sigma, R)

            remove_coords = np.array([
                [x, y] for x in range(shape[0]) for y in range(shape[1])
                if dist([x, y], [s.mu[1], s.mu[0]]) <= R
                   and data_field.field[x, y] > 0
            ])
            remove_coords = remove_coords.astype(int)

            for c in remove_coords:
                valid_coords_mask[c[0], c[1]] = True
            field = np.ma.array(
                data=data_field.field,
                mask=valid_coords_mask,
                fill_value=np.nan
            )
        else:
            print("-- Found background")
            for c in valid_coords:
                valid_coords_mask[c[0], c[1]] = True
            field = np.ma.array(
                data=data_field.field,
                mask=valid_coords_mask,
                fill_value=np.nan
            )
            continue

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


if __name__ == "__main__":
    main()
