import argparse as ag
import sys

import cpnest
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.special as ssp

from cpnest.model import Model
from pathlib import Path
from scipy.stats import multivariate_normal as mvn
from scipy.stats import poisson, norm

from pyfieldsim.core.fieldtypes.field import Field
from pyfieldsim.utils.metadata import read_metadata


def main():
    parser = ag.ArgumentParser(
        prog='fs-find-stars',
        description='',
    )
    parser.add_argument('data_file')
    parser.add_argument("-b", "--background-mean", type=float,
                        dest='b_mean', default=None, required=True,
                        help="")
    parser.add_argument("-l", "--limit", type=float,
                        dest='b_limit', default=None, required=True,
                        help="")
    parser.add_argument("-o", "--output",
                        dest='out_folder', default=None,
                        help="")

    args = parser.parse_args()

    if args.out_folder is None:
        out_folder = Path(os.getcwd())
    else:
        out_folder = Path(args.out_folder)

    data_file = Path(args.data_file)

    sources_file = Path('S' + data_file.name[1:])
    sources_metadata = read_metadata(sources_file)

    bkgnd_analysis_metadata = read_metadata(
        data_file.stem + '_bkg_analysis_meta.h5'
    )

    data_field = Field.from_field(data_file)
    shape = sources_metadata['shape']

    # 1 - Find the brightest star. Take all points in a circle around some
    # user-defined distance from the bright point (5 times the estimated psf
    # radius?).
    brt = data_field.field.max()
    brt_coords = np.unravel_index(np.argmax(data_field.field), shape)

    valid_coords = np.array([
        [x, y] for x in range(shape[0]) for y in range(shape[1])
        if dist([x, y], brt_coords) <= 15
    ])  # 15 may be arbitrary
    valid_counts = np.array([
        data_field.field[x[0], x[1]] for x in valid_coords
    ])

    # 2 - Set bounds for mean at +- 1 px from the brightest point. Set
    # bounds for sigma. NB: should be prior, can't watch the data!
    m_bounds = [
        [brt_coords[0] - 1, brt_coords[0] + 1],
        [brt_coords[1] - 1, brt_coords[1] + 1]
    ]
    s_bounds = [
        [0, 10]
    ]

    # 3 - Set bounds for brightness. From b_limit to max_value. A is surely
    # inside.
    b_limit = bkgnd_analysis_metadata['mean'] \
        + args.b_limit * bkgnd_analysis_metadata['std']
    A_bounds = [
        [0, 10 * brt]
    ]
    b_bounds = [
        [0, bkgnd_analysis_metadata['mean'] \
            + 3 * bkgnd_analysis_metadata['std']
         ]
    ]

    # 4 - Find one star within this radius.
    bounds = A_bounds + m_bounds + s_bounds + b_bounds

    fit_model_1 = FindPsf(
        valid_coords, valid_counts,
        (bkgnd_analysis_metadata['mean'], bkgnd_analysis_metadata['std']),
        bounds
    )

    psf1_path = Path('./sampling_output_psf1/')
    if not psf1_path.exists():  #
        work = cpnest.CPNest(
            fit_model_1,
            verbose=2,
            nlive=500,  # 1000
            maxmcmc=5000,  # 5000
            nensemble=4,
            output=psf1_path,
        )
        work.run()
        post_1 = work.posterior_samples.ravel()

        logZ_1 = work.logZ
    else:
        with h5py.File(psf1_path.joinpath('cpnest.h5'), 'r') as f:
            post_1 = np.asarray(f['combined']['posterior_samples'])

            logZ_1 = np.asarray(f['combined']['logZ']).reshape((1,))[0]

    logZ_psf = {
        '1': logZ_1
    }

    columns = [post_1[par] for par in post_1.dtype.names
               if par not in ['logL', 'logPrior']]
    samples = np.column_stack(columns)

    from corner import corner
    c = corner(
        samples,
        labels=[f'{par}' for par in post_1.dtype.names
                if par not in ['logL', 'logPrior']],
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
        bounds
    )

    psf2_path = Path('./sampling_output_psf2/')
    if not psf2_path.exists():
        work = cpnest.CPNest(
            fit_model_2,
            verbose=2,
            nlive=100,  # 1000
            maxmcmc=1000,  # 5000
            nensemble=1,
            output='./sampling_output_psf2/',
        )
        work.run()
        post_2 = work.posterior_samples.ravel()

        logZ_2 = work.logZ
    else:
        with h5py.File(psf2_path.joinpath('cpnest.h5'), 'r') as f:
            post_2 = np.asarray(f['combined']['posterior_samples'])

            logZ_2 = np.asarray(f['combined']['logZ']).reshape((1,))[0]

    logZ_psf['2'] = logZ_2

    columns = [post_2[par] for par in post_2.dtype.names
               if par not in ['logL', 'logPrior']]
    samples = np.column_stack(columns)

    c = corner(
        samples,
        labels=[f'{par}' for par in post_2.dtype.names
                if par not in ['logL', 'logPrior']],
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
        stars.append(
            new_star(
                A=np.median(post_1['A']),
                mu=[
                    np.median(post_1['mu_y']),
                    np.median(post_1['mu_x'])
                ],
                sigma=np.median(post_1['sigma']) * np.eye(2)
            )
        )
    else:
        sigma = np.median(post_2['sigma']) * np.eye(2)
        for _ in range(2):
            stars.append(
                new_star(
                    A=np.median(post_2[f'A{_}']),
                    mu=[
                        np.median(post_2[f'mu_y{_}']),
                        np.median(post_2[f'mu_x{_}'])
                    ],
                    sigma=sigma
                )
            )

    fig, ax = plt.subplots()
    ax.imshow(data_field.field, cmap='Greys', origin='upper')
    for s in stars:
        ax.scatter(s.mu[0], s.mu[1], marker='+', color='red')

    fig.savefig(
        out_folder.joinpath(f'recovered_psf_star.pdf'),
        bbox_inches='tight'
    )

    # 5 - Remove all points within 3s from the mean (or each mean) from the
    # dataset.
    reached_background = False


    # 6 - Find next brightest star and select all points in a circle around
    # 5σ from the bright point.

    # 7 - Check against background only, 1, 2 or 3 stars in the save
    # dataset. Compare evidences.

    # 8 - Remove all points within 3s from the mean (or each mean) from the
    # dataset.

    # 9 - Iterate from 6# until background term dominates in a dataset.

    # 10 - Profit.

    ## Forget, for now, what's below here
    # cleaned_data_coords = np.argwhere(
    #     data_field.field > b_limit
    # )
    # cleaned_data = np.asarray([
    #     data_field.field[c[0], c[1]] for c in cleaned_data_coords
    # ])
    # data = [
    #     [x[0], x[1], c] for x, c in zip(cleaned_data_coords, cleaned_data)
    # ]
    #
    # fit_model = StarField(
    #     data, 1, args.b_mean, data_field.field.shape
    # )
    #
    # work = cpnest.CPNest(
    #     fit_model,
    #     verbose=2,
    #     nlive=100,  # 1000
    #     maxmcmc=500,  # 5000
    #     nensemble=1,
    #     output='./sampling_output/',
    # )
    # work.run()
    # post = work.posterior_samples.ravel()


class FindPsf(Model):
    def __init__(self, coords, counts, background, bounds):
        self.coords = coords
        self.c = counts
        self.bkgnd = background[0]
        self.bkgnd_std = background[1]

        self.names = ['A', 'mu_x', 'mu_y', 'sigma', 'b']
        self.bounds = bounds

        self.n_pts = len(counts)

    def log_prior(self, param):
        log_p = super(FindPsf, self).log_prior(param)
        if np.isfinite(log_p):
            log_p = 0.
            log_p += norm.logpdf(param['b'],
                                 loc=self.bkgnd,
                                 scale=self.bkgnd_std)

        return log_p

    def log_likelihood(self, param):
        A = param['A']
        mu_x = param['mu_x']
        mu_y = param['mu_y']
        sigma = param['sigma']
        b = param['b']

        star = mvn(mean=[mu_x, mu_y], cov=sigma*np.eye(2))
        # c_hat = A * star.pdf(self.coords) + self.bkgnd
        c_hat = A * star.pdf(self.coords) + b

        likel = poisson.logpmf(self.c, c_hat)

        return likel.sum()


class FindPsf2(Model):
    def __init__(self, coords, counts, background, bounds):
        self.coords = np.asarray(coords).astype(int)
        self.c = np.asarray(counts).astype(int)
        self.bkgnd = background[0]
        self.bkgnd_std = background[1]

        self.names = [
            'A0', 'A1',
            'mu_x0', 'mu_y0',
            'mu_x1', 'mu_y1',
            'sigma', 'b'
        ]
        self.bounds = [
            bounds[0], bounds[0],
            bounds[1], bounds[1],
            bounds[1], bounds[1],
            bounds[2],
            bounds[3],
        ]

        self.n_pts = len(counts)

    def log_prior(self, param):
        log_p = super(FindPsf2, self).log_prior(param)

        if np.isfinite(log_p):
            log_p = 0.
            log_p += norm.logpdf(param['b'],
                                 loc=self.bkgnd,
                                 scale=self.bkgnd_std)

        return log_p

    def log_likelihood(self, param):
        A0 = param['A0']
        A1 = param['A1']
        mu_x0 = param['mu_x0']
        mu_y0 = param['mu_y0']
        mu_x1 = param['mu_x1']
        mu_y1 = param['mu_y1']
        sigma = param['sigma']
        b = param['b']

        star0 = mvn(mean=[mu_x0, mu_y0], cov=sigma * np.eye(2))
        star1 = mvn(mean=[mu_x1, mu_y1], cov=sigma * np.eye(2))
        c_hat = A0 * star0.pdf(self.coords) \
            + A1 * star1.pdf(self.coords) \
            + b

        likel = poisson.logpmf(self.c, c_hat)

        return likel.sum()


# class StarField(Model):
#     def __init__(self, data, n_stars, background, fieldsize):
#         self.x = np.asarray([d[0] for d in data])
#         self.y = np.asarray([d[1] for d in data])
#
#         self.coords = np.asarray([
#             [x, y] for x, y in zip(self.x, self.y)
#         ])
#         self.c = np.asarray([d[2] for d in data])
#
#         self.n_stars = n_stars
#         self.bkgnd = background
#         self.f_size = fieldsize
#
#         self.n_pts = len(data)
#
#         self.names = [
#             f'A_{n}' for n in range(self.n_stars)
#         ] + [
#             f'mux_{n}' for n in range(self.n_stars)
#         ] + [
#             f'muy_{n}' for n in range(self.n_stars)
#         ] + ['sigma']
#
#         self.bounds = [
#             [self.c.min() // 2, self.c.max()] for _ in range(self.n_stars)
#         ] + [
#             [self.x.min(), self.x.max()] for _ in range(self.n_stars)
#         ] + [
#             [self.y.min(), self.y.max()] for _ in range(self.n_stars)
#         ] + [
#             [0, 10]
#         ]
#
#     def log_prior(self, param):
#         log_p = super(StarField, self).log_prior(param)
#         if np.isfinite(log_p):
#             log_p = 0.
#
#         return log_p
#
#     def log_likelihood(self, param):
#         A = param.values[:self.n_stars]
#         mu_x = param.values[self.n_stars: 2 * self.n_stars]
#         mu_y = param.values[2 * self.n_stars: 3 * self.n_stars]
#         sigma = param['sigma']
#
#         star = mvn(mean=[0, 0], cov=sigma*np.eye(2))
#
#         centers = np.array([
#             [mx, my] for mx, my in zip(mu_x, mu_y)
#         ])
#         contributes = np.array([
#             a * star.pdf(
#                 self.coords - c0 * np.ones(
#                     (self.coords.shape[0], 2)
#                 )
#             )
#             for a, c0 in zip(A, centers)
#         ])
#
#         exp_c = contributes.sum(axis=0) + self.bkgnd * np.ones(len(self.x))
#         likel = self.c * exp_c - exp_c
#         likel -= ssp.gammaln(self.c + np.ones(len(self.c)))
#
#         return likel.sum()


def dist(x1, x2):
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)

    d = np.sqrt(((x2 - x1) ** 2).sum())
    return d


class Star:
    def __init__(self, A, mu, sigma):
        self.A = A
        self.mu = mu
        self.sigma = sigma

        self.dist = mvn(
            mean=[self.mu[0], self.mu[1]],
            cov=self.sigma * np.eye(2))

    def __call__(self, x):
        return self.A * mvn.pdf(x)


def new_star(A, mu, sigma):
    return Star(A, mu, sigma)


if __name__ == "__main__":
    main()
