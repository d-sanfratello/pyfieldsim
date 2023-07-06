import numpy as np

from scipy.stats import multivariate_normal as mvn

from point_star import PointStar


class Star(PointStar):
    def __init__(self, A, mu, sigma):
        super(Star, self).__init__(A, mu)

        self.sigma = sigma

        self.dist = mvn(
            mean=[self.mu[0], self.mu[1]],
            cov=self.sigma * np.eye(2))

    def __call__(self, x):
        return self.A * mvn.pdf(x)


def new_star(A, mu, sigma):
    return Star(A, mu, sigma)
