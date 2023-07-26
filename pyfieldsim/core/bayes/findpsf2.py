import numpy as np

from cpnest.model import Model
from scipy.stats import multivariate_normal as mvn
from scipy.stats import poisson, norm


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
            self.log_likelihood = self.__log_l_flat
        else:
            self.bkgnd = background[0]
            self.bkgnd_std = background[1]
            self.log_likelihood = self.__log_l_bgnd

        self.n_pts = len(counts)

    def log_prior(self, param):
        log_p = super(FindPsf2, self).log_prior(param)

        if np.isfinite(log_p):
            log_p = 0.
            if not self.is_flat:
                log_p += norm.logpdf(param['b'],
                                     loc=self.bkgnd,
                                     scale=self.bkgnd_std)
                # log_p -= np.log(param['b'])

        return log_p

    def __log_l_flat(self, param):
        A0 = param['A0']
        A1 = param['f'] * A0
        mu_x0 = param['mu_x0']
        mu_y0 = param['mu_y0']
        mu_x1 = param['mu_x1']
        mu_y1 = param['mu_y1']
        sigma = param['sigma']

        stars_0_cts = A0 * norm.pdf(
            self.coords[:, 0], loc=mu_x0, scale=sigma
        ) * norm.pdf(
            self.coords[:, 1], loc=mu_y0, scale=sigma
        )
        stars_1_cts = A1 * norm.pdf(
            self.coords[:, 0], loc=mu_x1, scale=sigma
        ) * norm.pdf(
            self.coords[:, 1], loc=mu_y1, scale=sigma
        )
        c_hat = stars_0_cts + stars_1_cts

        likel = poisson.logpmf(self.c, c_hat)

        return likel.sum()

    def __log_l_bgnd(self, param):
        A0 = param['A0']
        A1 = param['f'] * A0
        mu_x0 = param['mu_x0']
        mu_y0 = param['mu_y0']
        mu_x1 = param['mu_x1']
        mu_y1 = param['mu_y1']
        sigma = param['sigma']
        b = param['b']

        stars_0_cts = A0 * norm.pdf(
            self.coords[:, 0], loc=mu_x0, scale=sigma
        ) * norm.pdf(
            self.coords[:, 1], loc=mu_y0, scale=sigma
        )
        stars_1_cts = A1 * norm.pdf(
            self.coords[:, 0], loc=mu_x1, scale=sigma
        ) * norm.pdf(
            self.coords[:, 1], loc=mu_y1, scale=sigma
        )
        c_hat = stars_0_cts + stars_1_cts + b

        likel = poisson.logpmf(self.c, c_hat)

        return likel.sum()
