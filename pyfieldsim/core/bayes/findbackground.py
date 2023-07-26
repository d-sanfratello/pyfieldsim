import numpy as np

from cpnest.model import Model
from scipy.stats import multivariate_normal as mvn
from scipy.stats import poisson, norm


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
            # log_p -= np.log(param['b'])

        return log_p

    def log_likelihood(self, param):
        b = param['b']

        c_hat = b * np.ones(shape=self.c.shape)
        likel = poisson.logpmf(self.c, c_hat)

        return likel.sum()
