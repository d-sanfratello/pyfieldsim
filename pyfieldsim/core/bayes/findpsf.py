import numpy as np

from cpnest.model import Model
from scipy.stats import poisson, norm


class FindPsf(Model):
    def __init__(self, coords, counts,
                 background_meta,
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
            self.bkgnd = background_meta['mean']
            self.bkgnd_std = background_meta['std']
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

        return log_p

    def __log_l_flat(self, param):
        # likelihood for the flat case, without any background.

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
        # likelihood for the background case.

        A = param['A']
        mu_x = param['mu_x']
        mu_y = param['mu_y']
        sigma = param['sigma']
        b = param['b']

        # expected counts from a single star
        star_cts = A * norm.pdf(
            self.coords[:, 0],
            loc=mu_x, scale=sigma
        ) * norm.pdf(
            self.coords[:, 1],
            loc=mu_y, scale=sigma
        )
        c_hat = star_cts + b

        if self.c.dtype == int:
            likel = poisson.logpmf(self.c, c_hat)
        else:
            # for continuous variables, like after RL deconvolution
            likel = norm.logpdf(self.c,
                                loc=c_hat,
                                scale=np.sqrt(c_hat))

        return likel.sum()
