import numpy as np

from cpnest.model import Model
from scipy.stats import poisson, norm


class FindStar(Model):
    def __init__(self, coords, counts,
                 background_meta,
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
            self.log_likelihood = self.__log_l_flat
        else:
            self.bkgnd = background_meta['mean']
            self.bkgnd_std = background_meta['std']
            self.log_likelihood = self.__log_l_bgnd

        self.sigma = sigma

        self.n_pts = len(counts)

    def log_prior(self, param):
        log_p = super(FindStar, self).log_prior(param)
        if np.isfinite(log_p):
            log_p = 0.
            if not self.is_flat:
                log_p += norm.logpdf(param['b'],
                                     loc=self.bkgnd,
                                     scale=self.bkgnd_std)

        return log_p

    def __log_l_flat(self, param):
        A = param['A']
        mu_x = param['mu_x']
        mu_y = param['mu_y']

        star_cts = A * norm.pdf(
            self.coords[:, 0], loc=mu_x, scale=self.sigma
        ) * norm.pdf(
            self.coords[:, 1], loc=mu_y, scale=self.sigma
        )

        likel = poisson.logpmf(self.c, star_cts)

        return likel.sum()

    def __log_l_bgnd(self, param):
        A = param['A']
        mu_x = param['mu_x']
        mu_y = param['mu_y']
        b = param['b']

        star_cts = A * norm.pdf(
            self.coords[:, 0], loc=mu_x, scale=self.sigma
        ) * norm.pdf(
            self.coords[:, 1], loc=mu_y, scale=self.sigma
        )
        c_hat = star_cts + b

        likel = poisson.logpmf(self.c, c_hat)

        return likel.sum()
