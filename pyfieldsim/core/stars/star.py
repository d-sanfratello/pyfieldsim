import numpy as np

from scipy.stats import multivariate_normal as mvn

from .point_star import PointStar


class Star(PointStar):
    def __init__(self, A, mu, sigma, fmts=None, pos_error=None, A_error=None):
        super(Star, self).__init__(A, mu)

        self.sigma = sigma

        self.dist = mvn(
            mean=[self.mu[0], self.mu[1]],
            cov=self.sigma * np.eye(2))

        self.__fmts = fmts
        self.__pos_error = [
            [pos_error[0], pos_error[1]],
            [pos_error[2], pos_error[3]]
        ]
        self.__A_error = A_error

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

    @property
    def pos_error(self):
        return self.__pos_error

    @property
    def A_error(self):
        return self.__A_error


def new_star(A, mu, sigma, fmts=None, pos_error=None, A_error=None):
    return Star(A, mu, sigma, fmts=fmts, pos_error=pos_error, A_error=A_error)
