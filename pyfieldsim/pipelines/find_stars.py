import argparse as ag
import sys

import cpnest
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

from corner import corner
from cpnest.model import Model
from pathlib import Path
from scipy.stats import multivariate_normal as mvn
from scipy.stats import poisson, norm

from pyfieldsim.core.fieldtypes.field import Field
from pyfieldsim.core.stars import new_star
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
        if s[0] >= 0 and s[0] < shape[0] and s[1] >= 0 and s[1] < shape[1]:
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
    # user-defined distance from the bright point (5 times the estimated psf
    # radius?).
    brt = data_field.field.max()
    brt_coords = np.unravel_index(np.argmax(data_field.field), shape)

    valid_coords = np.array([
        [x, y] for x in range(shape[0]) for y in range(shape[1])
        if dist([x, y], brt_coords) <= args.initial_width and
           data_field.field[x, y] > 0
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
        [0, 100 * brt]
    ]
    b_bounds = [[
        0,
        bkgnd_analysis_metadata['mean'] + 3 * bkgnd_analysis_metadata['std']
    ]]

    # 4 - Find one star within this radius.
    bounds = A_bounds + m_bounds + s_bounds + b_bounds

    fit_model_1 = FindPsf(
        valid_coords, valid_counts,
        (bkgnd_analysis_metadata['mean'], bkgnd_analysis_metadata['std']),
        bounds,
        is_flat=args.is_flat
    )

    psf1_path = Path('./sampling_output_psf1/')
    print("- Testing 1 star psf")
    if not psf1_path.exists():
        print("-- Inference run")
        work = cpnest.CPNest(
            fit_model_1,
            verbose=1,
            nlive=500,  # 1000
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

    columns_1 = [post_1[par] for par in post_1.dtype.names
               if par not in ['logL', 'logPrior']]
    samples_1 = np.column_stack(columns_1)
    labels = [f'${par}$' for par in post_1.dtype.names
              if par not in ['logL', 'logPrior']]
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
    )
    c.savefig(
        out_folder.joinpath(f'joint_posterior_psf1.pdf'),
        bbox_inches='tight'
    )

    # 4b - Check against two or three gaussians in the same dataset. Compare
    # evidences.
    bounds = A_bounds + m_bounds + s_bounds + b_bounds

    fit_model_2 = FindPsf2(
        valid_coords, valid_counts,
        (bkgnd_analysis_metadata['mean'], bkgnd_analysis_metadata['std']),
        bounds,
        is_flat=args.is_flat
    )

    psf2_path = Path('./sampling_output_psf2/')
    print("- Testing 2 stars psf")
    if not psf2_path.exists():
        print("-- Inference run")
        work = cpnest.CPNest(
            fit_model_2,
            verbose=1,
            nlive=500,  # 1000
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
    )
    c.savefig(
        out_folder.joinpath(f'joint_posterior_psf2.pdf'),
        bbox_inches='tight'
    )

    stars = []
    if logZ_psf['1'] >= logZ_psf['2']:
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
        print(f"--- Star at ({np.median(post_1['mu_y'])}, "
              f"{np.median(post_1['mu_x'])}), "
              f"of brightness {np.median(post_1['A'])}")
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
        print(f"--- Star at ({np.median(post_2[f'mu_y0'])}, "
              f"{np.median(post_2[f'mu_x0'])}), "
              f"of brightness {np.median(post_2[f'A0'])}")

        A1 = np.median(post_2['f'] * post_2['A0'])
        print(f"--- Star at ({np.median(post_2[f'mu_y1'])}, "
              f"{np.median(post_2[f'mu_x1'])}), "
              f"of brightness {A1}")

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
        out_folder.joinpath(f'recovered_psf_star.pdf'),
        bbox_inches='tight'
    )

    # 5 - Remove all points within 3s from the mean (or each mean) from the
    # dataset.
    # TODO: set min(3s, R) with R from background (see calcs)
    valid_coords_mask = np.zeros(shape).astype(bool)
    remove_coords = np.array([
        [x, y] for x in range(shape[0]) for y in range(shape[1])
        if dist([x, y], brt_coords) <= 1.5 * sigma
        and data_field.field[x, y] > 0
    ])
    for c in remove_coords:
        valid_coords_mask[c[0], c[1]] = True
    field = np.ma.array(
        data=data_field.field,
        mask=valid_coords_mask,
        fill_value=np.nan
    )


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
            [0, bkgnd_analysis_metadata['mean']
             + 3 * bkgnd_analysis_metadata['std']
             ]
        ]

        bounds = A_bounds + m_bounds + b_bounds

        fit_model_s = FindStar(
            valid_coords, valid_counts,
            # (bkgnd_analysis_metadata['mean'], bkgnd_analysis_metadata['std']),
            bounds,
            sigma,
            is_flat=args.is_flat
        )

        star_path = Path('./sampling_output_star/')
        print("-- Testing against star hypothesis")
        work = cpnest.CPNest(
            fit_model_s,
            verbose=0,
            nlive=100,  # 1000
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
                # (bkgnd_analysis_metadata['mean'], bkgnd_analysis_metadata['std']),
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


class FindPsf(Model):
    def __init__(self, coords, counts,
                 background,
                 bounds, is_flat=False):
        self.coords = coords
        self.c = counts

        self.names = ['A', 'mu_x', 'mu_y', 'sigma', 'b']
        self.bounds = bounds

        self.is_flat = is_flat
        if self.is_flat:
            self.names = self.names[:-1]
            self.bounds = self.bounds[:-1]
        else:
            self.bkgnd = background[0]
            self.bkgnd_std = background[1]

        self.n_pts = len(counts)

    def log_prior(self, param):
        log_p = super(FindPsf, self).log_prior(param)
        if np.isfinite(log_p):
            log_p = 0.

            if not self.is_flat:
                # log_p += norm.logpdf(param['b'],
                #                      loc=self.bkgnd,
                #                      scale=self.bkgnd_std)
                log_p -= np.log(param['b'])

        return log_p

    def log_likelihood(self, param):
        A = param['A']
        mu_x = param['mu_x']
        mu_y = param['mu_y']
        sigma = param['sigma']
        if not self.is_flat:
            b = param['b']
        else:
            b = 0

        star = mvn(mean=[mu_x, mu_y], cov=(sigma**2)*np.eye(2))
        c_hat = A * star.pdf(self.coords) + b

        likel = poisson.logpmf(self.c, c_hat)

        return likel.sum()


class FindPsf2(Model):
    def __init__(self, coords, counts,
                 background,
                 bounds, is_flat=False):
        self.coords = np.asarray(coords).astype(int)
        self.c = np.asarray(counts).astype(int)

        self.names = [
            'A0', 'f',
            'mu_x0', 'mu_y0',
            'mu_x1', 'mu_y1',
            'sigma', 'b'
        ]
        self.bounds = [
            bounds[0], [0, 1],
            bounds[1], bounds[2],
            bounds[1], bounds[2],
            bounds[3],
            bounds[4],
        ]

        self.is_flat = is_flat
        if is_flat:
            self.names = self.names[:-1]
            self.bounds = self.bounds[:-1]
        # else:
        #     self.bkgnd = background[0]
        #     self.bkgnd_std = background[1]

        self.n_pts = len(counts)

    def log_prior(self, param):
        log_p = super(FindPsf2, self).log_prior(param)

        if np.isfinite(log_p):
            log_p = 0.
            if not self.is_flat:
                # log_p += norm.logpdf(param['b'],
                #                      loc=self.bkgnd,
                #                      scale=self.bkgnd_std)
                if not self.is_flat:
                    log_p -= np.log(param['b'])

        return log_p

    def log_likelihood(self, param):
        A0 = param['A0']
        A1 = param['f'] * A0
        mu_x0 = param['mu_x0']
        mu_y0 = param['mu_y0']
        mu_x1 = param['mu_x1']
        mu_y1 = param['mu_y1']
        sigma = param['sigma']
        if self.is_flat:
            b = 0
        else:
            b = param['b']

        star0 = mvn(mean=[mu_x0, mu_y0], cov=(sigma**2) * np.eye(2))
        star1 = mvn(mean=[mu_x1, mu_y1], cov=(sigma**2) * np.eye(2))
        c_hat = A0 * star0.pdf(self.coords) \
            + A1 * star1.pdf(self.coords) \
            + b

        likel = poisson.logpmf(self.c, c_hat)

        return likel.sum()


class FindStar(Model):
    def __init__(self, coords, counts,
                 # background,
                 bounds, sigma,
                 is_flat=False):
        self.coords = coords
        self.c = counts

        self.names = ['A', 'mu_x', 'mu_y', 'b']
        self.bounds = bounds

        self.is_flat = is_flat
        if self.is_flat:
            self.names = self.names[:-1]
            self.bounds = self.bounds[:-1]
        # else:
        #     self.bkgnd = background[0]
        #     self.bkgnd_std = background[1]

        self.sigma = sigma

        self.n_pts = len(counts)

    def log_prior(self, param):
        log_p = super(FindStar, self).log_prior(param)
        if np.isfinite(log_p):
            log_p = 0.
            # if not self.is_flat:
            #     log_p += norm.logpdf(param['b'],
            #                          loc=self.bkgnd,
            #                          scale=self.bkgnd_std)
            if not self.is_flat:
                log_p -= np.log(param['b'])

        return log_p

    def log_likelihood(self, param):
        A = param['A']
        mu_x = param['mu_x']
        mu_y = param['mu_y']
        if self.is_flat:
            b = 0
        else:
            b = param['b']

        star = mvn(mean=[mu_x, mu_y], cov=(self.sigma**2) * np.eye(2))
        c_hat = A * star.pdf(self.coords) + b

        likel = poisson.logpmf(self.c, c_hat)

        return likel.sum()


class FindBackground(Model):
    def __init__(self, coords, counts,
                 # background,
                 bounds):
        self.coords = coords
        self.c = counts
        # self.bkgnd = background[0]
        # self.bkgnd_std = background[1]

        self.names = ['b']
        self.bounds = bounds

        self.n_pts = len(counts)

    def log_prior(self, param):
        log_p = super(FindBackground, self).log_prior(param)
        if np.isfinite(log_p):
            log_p = 0
            # log_p += norm.logpdf(param['b'],
            #                      loc=self.bkgnd,
            #                      scale=self.bkgnd_std)
            log_p -= np.log(param['b'])

        return log_p

    def log_likelihood(self, param):
        b = param['b']

        c_hat = b * np.ones(shape=self.c.shape)
        likel = poisson.logpmf(self.c, c_hat)

        return likel.sum()


def dist(x1, x2):
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)

    d = np.sqrt(((x2 - x1) ** 2).sum())
    return d


if __name__ == "__main__":
    main()
