import numpy as np

from scipy.stats import multivariate_normal as mvn

from .point_star import PointStar


class Star(PointStar):
    def __init__(self, A, mu, sigma, fmts):
        super(Star, self).__init__(A, mu)

        self.sigma = sigma

        self.dist = mvn(
            mean=[self.mu[0], self.mu[1]],
            cov=self.sigma * np.eye(2))

        self.__fmts = fmts

    def __call__(self, x):
        return self.A * mvn.pdf(x)

    @property
    def fmt_A(self):
        return self.__fmts[0]

    @property
    def fmt_mu_x(self):
        return self.__fmts[1]

    @property
    def fmt_mu_y(self):
        return self.__fmts[2]


def new_star(A, mu, sigma, fmts):
    return Star(A, mu, sigma, fmts)
