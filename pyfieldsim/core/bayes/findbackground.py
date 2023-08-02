import numpy as np

from cpnest.model import Model
from scipy.stats import poisson, norm


class FindBackground(Model):
    def __init__(self, coords, counts,
                 background_meta,
                 bounds):
        self.coords = coords
        self.c = counts
        self.bkgnd = background_meta['mean']
        self.bkgnd_std = background_meta['std']

        self.names = ['b']
        self.bounds = bounds

        self.n_pts = len(counts)

    def log_prior(self, param):
        log_p = super(FindBackground, self).log_prior(param)
        if np.isfinite(log_p):
            log_p = 0
            log_p += norm.logpdf(param['b'],
                                 loc=self.bkgnd,
                                 scale=self.bkgnd_std)

        return log_p

    def log_likelihood(self, param):
        b = param['b']

        c_hat = b * np.ones(shape=self.c.shape)
        likel = poisson.logpmf(self.c, c_hat)

        return likel.sum()
