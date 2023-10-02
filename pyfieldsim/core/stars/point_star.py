import numpy as np


class PointStar:
    """
    Class to define a delta-like star, containing the brightness and the
    centroid position.
    """
    def __init__(self, A, mu):
        self.A = A
        self.mu = mu

    def __call__(self, x):
        if np.equal(np.round(x), self.mu):
            return self.A
        else:
            return 0


def new_point_star(A, mu):
    """
    Function that generates a new `PointStar` for given brightness and
    position.

    Parameters
    ----------
    A: `number`
        The brightness of the star
    mu: `numpy.ndarray` of shape (2,)
        The centroid position of the star.

    Returns
    -------
    PointStar
        An instance of a new `PointStar`.
    """
    return PointStar(A, mu)
