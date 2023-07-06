import numpy as np


class PointStar:
    def __init__(self, A, mu):
        self.A = A
        self.mu = mu

    def __call__(self, x):
        if np.equal(np.round(x), self.mu):
            return self.A
        else:
            return 0


def new_point_star(A, mu):
    return PointStar(A, mu)
