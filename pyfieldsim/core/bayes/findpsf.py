import numpy as np

from cpnest.model import Model
from scipy.stats import multivariate_normal as mvn
from scipy.stats import poisson, norm


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
            self.log_likelihood = self.__log_l_flat
        else:
            self.bkgnd = background[0]
            self.bkgnd_std = background[1]
            self.log_likelihood = self.__log_l_bgnd

        self.n_pts = len(counts)

    def log_prior(self, param):
        log_p = super(FindPsf, self).log_prior(param)
        if np.isfinite(log_p):
            log_p = 0.

            if not self.is_flat:
                log_p += norm.logpdf(param['b'],
                                     loc=self.bkgnd,
                                     scale=self.bkgnd_std)
                # log_p -= np.log(param['b'])

        return log_p

    def __log_l_flat(self, param):
        A = param['A']
        mu_x = param['mu_x']
        mu_y = param['mu_y']
        sigma = param['sigma']

        star_cts = A * norm.pdf(
            self.coords[:, 0],
            loc=mu_x, scale=sigma
        ) * norm.pdf(
            self.coords[:, 1],
            loc=mu_y, scale=sigma
        )

        likel = poisson.logpmf(self.c, star_cts)

        return likel.sum()

    def __log_l_bgnd(self, param):
        A = param['A']
        mu_x = param['mu_x']
        mu_y = param['mu_y']
        sigma = param['sigma']
        b = param['b']

        star_cts = A * norm.pdf(
            self.coords[:, 0],
            loc=mu_x, scale=sigma
        ) * norm.pdf(
            self.coords[:, 1],
            loc=mu_y, scale=sigma
        )
        c_hat = star_cts + b

        likel = poisson.logpmf(self.c, c_hat)

        return likel.sum()