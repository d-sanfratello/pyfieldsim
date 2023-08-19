import argparse as ag

import h5py
import numpy as np
import os

from pathlib import Path
from shutil import rmtree

from pyfieldsim.core.fieldtypes.field import Field
from pyfieldsim.core.stars.find_utils import (
    dist,
    mask_field,
    median_quantiles,
    plot_recovered_stars,
    run_mcmc,
    select_hypothesis,
    select_valid_pixels,
)

from pyfieldsim.core.bayes import FindPsf, FindPsf2
from pyfieldsim.core.bayes import FindStar, FindBackground

from pyfieldsim.utils.metadata import read_metadata
from pyfieldsim.utils.save_stars import save_stars


# noinspection PyArgumentList,PyUnboundLocalVariable
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
    parser.add_argument("--no-force", action='store_false',
                        dest='no_force', default=True,
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

    plot_folder.mkdir(parents=True, exist_ok=True)
    samplings_folder.mkdir(parents=True, exist_ok=True)

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
        limit_points = 4
    else:
        bkgnd_analysis_metadata = {
            'mean': 0,
            'std': 1
        }
        limit_points = 3

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
    )[:-1]
    logZ_psf = {'1': logZ_1}
    if not args.is_flat:
        bgnd_50, err_m, err_p = median_quantiles(post_1['b'], cl=2)[:-1]
        bgnd_5_95 = {'1': [bgnd_50 - err_m, bgnd_50 + err_p]}

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
        )[:-1]

    # If the two stars lies farther than the initial radius from each other,
    # the two stars hypothesis is discarded.
    try:
        median_0 = (
            np.median(post_2['mu_x0']),
            np.median(post_2['mu_y0'])
        )
        median_1 = (
            np.median(post_2['mu_x1']),
            np.median(post_2['mu_y1'])
        )
        s0_s1_dist = dist(median_0, median_1)
    except IndexError:
        pass
    else:
        if s0_s1_dist > initial_radius:
            logZ_2 = -np.inf

    logZ_psf['2'] = logZ_2
    if not args.is_flat:
        bgnd_50, err_m, err_p = median_quantiles(post_2['b'], cl=2)[:-1]
        # noinspection PyUnboundLocalVariable
        bgnd_5_95['2'] = [bgnd_50 - err_m, bgnd_50 + err_p]

    stars = []
    pos_errors = []

    # 10^0.5 -> substantial evidence
    # 10^1 -> strong evidence
    # 10^1.5 -> very strong evidence
    # 10^2 -> decisive evidence
    sigma, b_u, hyp_psf = select_hypothesis(
        hyp_1='1', hyp_2='2',
        logZ=logZ_psf,
        logB_lim_hyp2=-1 * np.log(10),
        stars=stars,
        pos_errors=pos_errors,
        post_1=post_1,
        post_2=post_2,
        is_flat=args.is_flat,
    )

    plot_recovered_stars(
        data_field.field,
        stars=stars,
        pos_errors=pos_errors,
        shape=shape,
        show_sources=args.show_sources,
        is_flat=args.is_flat,
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
        sigma=sigma, b_u=b_u,
        is_flat=args.is_flat
    )

    plot_recovered_stars(
        data_field.field,
        stars=stars,
        pos_errors=pos_errors,
        shape=shape,
        out_path=plot_folder.joinpath(f'removed_pixels_psf_star.pdf'),
        show_sources=True,
        is_flat=args.is_flat,
        sources=coords,
        brt_coords=brt_coords,
        radius=initial_radius,
        show_mask=True,
        masked_field=field,
        sigma=sigma,
        b_u=b_u
    )

    # 6 - Find next brightest star and select all points in a circle around
    # 5σ from the bright point.
    exit_loop = False
    star_id = len(stars) + 1
    saved_ids = [_ + 1 for _ in range(len(stars))]

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

        radius = 2 * sigma
        valid_coords, valid_counts = select_valid_pixels(
            field,
            radius=radius,
            shape=shape,
            brt_coords=brt_coords
        )

        if valid_coords.shape[0] <= limit_points + 1:
            field, valid_coords_mask = mask_field(
                field,
                stars=stars,
                mask=valid_coords_mask,
                shape=shape,
                sigma=sigma, b_u=b_u,
                is_flat=args.is_flat,
                force_remove=valid_coords
            )
            print("-- Too few points to infer a star")
            continue

        # 7 - Check against background only or 1 star in the dataset.
        # Compare evidences.
        mins = np.min(valid_coords, axis=0)
        maxs = np.max(valid_coords, axis=0)
        m_bounds = [
            [mins[0], maxs[0]],
            [mins[1], maxs[1]]
        ]
        A_bounds = [
            [brt / 10, 100 * brt]
        ]
        # b_bounds = [[
        #     max(
        #         bkgnd_analysis_metadata['mean']
        #         - 2 * bkgnd_analysis_metadata['std'], 1),
        #     bkgnd_analysis_metadata['mean'] + bkgnd_analysis_metadata['std']
        # ]]
        b_bounds = [
            [0, 1]
        ]
        if not args.is_flat:
            b_bounds = [bgnd_5_95[hyp_psf]]

        bounds = A_bounds + m_bounds + b_bounds

        fit_model_s = FindStar(
            valid_coords, valid_counts,
            bkgnd_analysis_metadata,
            bounds,
            sigma,
            is_flat=args.is_flat
        )

        print("-- Testing against star hypothesis")
        post_s, logZ_s, path_s = run_mcmc(
            model=fit_model_s,
            name=f'star_{star_id}',
            out_folder=plot_folder,
            verbose=1,
            force=args.no_force
        )
        logZ = {'s': logZ_s}

        if not args.is_flat:
            print("-- Testing against pure background hypothesis")
            bounds = b_bounds
            fit_model_b = FindBackground(
                valid_coords, valid_counts,
                bkgnd_analysis_metadata,
                bounds
            )

            post_b, logZ_b, path_b = run_mcmc(
                model=fit_model_b,
                name=f'bgnd_{star_id}',
                out_folder=plot_folder,
                verbose=1,
                force=args.no_force
            )
        else:
            logZ_b = -np.inf
            post_b = None

        # If bright point (in recovery) lies farther than 2 sigma from the
        # saved mean, it means we are analyzing a different region.
        # if dist(
        #         brt_coords,
        #         (np.median(post_s['mu_y']), np.median(post_s['mu_x']))
        # ) >= radius and not args.is_flat:
        #     logZ_b = +np.inf
        #
        # for s in stars:
        #     s_ = (np.median(post_s['mu_y']), np.median(post_s['mu_x']))
        #     if dist((s.mu[0], s.mu[1]),
        #             s_) < sigma:
        #         logZ_b = +np.inf
        #         break

        logZ['b'] = logZ_b

        # noinspection PyUnboundLocalVariable
        sigma, b_u, hyp_s_b = select_hypothesis(
            hyp_1='s', hyp_2='b',
            logZ=logZ,
            logB_lim_hyp2=0.5 * np.log(10),
            stars=stars,
            pos_errors=pos_errors,
            post_1=post_s,
            post_2=post_b,
            is_flat=args.is_flat,
            sigma=sigma
        )

        plot_recovered_stars(
            data_field.field,
            stars=stars,
            pos_errors=pos_errors,
            shape=shape,
            show_sources=args.show_sources,
            is_flat=args.is_flat,
            sources=coords,
            brt_coords=brt_coords,
            radius=radius,
            out_path=plot_folder.joinpath(f'recovered_{star_id}.pdf')
        )

        if hyp_s_b == 's':
            saved_ids.append(star_id)
        elif hyp_s_b == 'b' and not args.no_force:
            if path_s.exists():
                rmtree(path_s, ignore_errors=True)
            if path_b.exists():
                rmtree(path_b, ignore_errors=True)

        # 8 - Remove all points within R from the mean from the dataset.
        if not args.is_flat and not np.isnan(b_u):
            field, valid_coords_mask = mask_field(
                field,
                stars=stars,
                mask=valid_coords_mask,
                shape=shape,
                sigma=sigma, b_u=b_u,
                is_flat=args.is_flat
            )

            plot_recovered_stars(
                data_field.field,
                stars=stars,
                pos_errors=pos_errors,
                shape=shape,
                out_path=plot_folder.joinpath(f'removed_pixels_star'
                                              f'{star_id}.pdf'),
                show_sources=True,
                is_flat=args.is_flat,
                sources=coords,
                brt_coords=brt_coords,
                radius=radius,
                show_mask=True,
                masked_field=field,
                sigma=sigma,
                b_u=b_u
            )
        elif args.is_flat or np.isnan(b_u):
            field, valid_coords_mask = mask_field(
                field,
                stars=stars,
                mask=valid_coords_mask,
                shape=shape,
                sigma=sigma, b_u=b_u,
                is_flat=args.is_flat,
                force_remove=valid_coords
            )

            plot_recovered_stars(
                data_field.field,
                stars=stars,
                pos_errors=pos_errors,
                shape=shape,
                out_path=plot_folder.joinpath(f'removed_pixels_star'
                                              f'{star_id}.pdf'),
                show_sources=True,
                is_flat=args.is_flat,
                sources=coords,
                brt_coords=brt_coords,
                radius=radius,
                show_mask=True,
                masked_field=field,
                sigma=sigma,
                b_u=b_u
            )

            if args.is_flat and hyp_s_b == 's':
                star_id += 1
            continue

        # 9 - Iterate from 6# until background term dominates in a dataset.
        star_id += 1

    if args.options is not None:
        options = args.options
    else:
        options = ''
    save_stars(stars, data_file, saved_ids, hyp_psf,
               options=options)

    # Correlating stars between the first (fifo) and the ones within 2 sigmas
    print("- Removing aliases")
    if args.is_flat:
        n_limit = 3
    else:
        n_limit = 2

    if hyp_psf == '1':
        psf_stars = stars[:1]

        analized_stars_idx = [
            _+1 for _, s in enumerate(stars[1:])
            if dist(s.mu, psf_stars[0].mu) <= n_limit * sigma
        ]
    else:
        psf_stars = stars[:2]

        analized_stars_idx = [
            _+2 for _, s in enumerate(stars[2:])
            if dist(s.mu, psf_stars[0].mu) <= n_limit * sigma
            or dist(s.mu, psf_stars[1].mu) <= n_limit * sigma
        ]

    if len(analized_stars_idx) > 1:
        psf_stars = np.array([
            p.mu for p in psf_stars
        ])
        analized_stars = np.array([
            stars[_].mu for _ in analized_stars_idx
        ])

        mean_analized_stars = np.mean(analized_stars, axis=0)

        for p in psf_stars:
            if dist(p, mean_analized_stars) <= sigma:
                for idx in sorted(analized_stars_idx, reverse=True):
                    del stars[idx]
                    del saved_ids[idx]
                    del pos_errors[idx]
                break

    ctr = len(psf_stars)
    remaining_stars = stars[ctr:]
    while len(remaining_stars) > 0:
        ref_star = remaining_stars[0]
        ctr += 1

        analized_stars_idx = [
            _ + ctr for _, s in enumerate(stars[ctr:])
            if dist(s.mu, ref_star.mu) <= n_limit * sigma
        ]
        if len(analized_stars_idx) > 1:
            analized_stars = np.array([
                stars[_].mu for _ in analized_stars_idx
            ])
            mean_analized_stars = np.mean(analized_stars, axis=0)

            if dist(ref_star.mu, mean_analized_stars) <= sigma:
                for idx in sorted(analized_stars_idx, reverse=True):
                    del stars[idx]
                    del saved_ids[idx]
                    del pos_errors[idx]

        remaining_stars = stars[ctr:]

    # 10 - Profit.
    plot_recovered_stars(
        data_field.field,
        stars=stars,
        pos_errors=pos_errors,
        shape=shape,
        show_sources=args.show_sources,
        is_flat=args.is_flat,
        sources=coords,
        out_path=plot_folder.joinpath(f'recovered_stars_all.pdf')
    )

    plot_recovered_stars(
        data_field.field,
        stars=stars,
        pos_errors=pos_errors,
        shape=shape,
        out_path=plot_folder.joinpath(f'removed_pixels_all.pdf'),
        show_sources=True,
        is_flat=args.is_flat,
        sources=coords,
        show_mask=True,
        masked_field=field,
        sigma=sigma,
        b_u=b_u
    )

    save_stars(stars, data_file, saved_ids, hyp_psf, options=args.options)


if __name__ == "__main__":
    main()
