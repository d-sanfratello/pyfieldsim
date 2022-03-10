import numpy as np

from psf import Kernel


class GaussKernel(Kernel):
    def __init__(self, sigma, size=2.5):
        """
        Class that initializates a gaussian psf kernel with central symmetry.

        It defines a square kernel array with axis of len `int(size * sigma)`, where `sigma` is the standard
        deviation of the distribution. The kernel is normalized.

        Parameters
        ----------
        sigma:
            The standard deviation of a 2D gaussian distribution with central symmetry.
        size:
            The multiplier to obtain the size of the kernel. In this case it corresponds to `size` sigmas.
            Default is `2.5`.
        """
        super(GaussKernel, self).__init__('gauss', sigma, sigma, size)

    def _generate_kernel(self):
        x_axis = np.arange(self.size_x) - self.size_x // 2
        y_axis = np.arange(self.size_y) - self.size_y // 2

        xx, yy = np.meshgrid(x_axis, y_axis, sparse=True)

        return np.exp(-0.5 * (xx**2 + yy**2)) / (2 * np.pi * self.width_x * self.width_y)


class AsymmGaussKernel(Kernel):
    def __init__(self, sigma_x, sigma_y, size=2.5):
        """
        Class that initializates a gaussian psf kernel with axis symmetries.

        It defines a rectangular kernel array of shape `(int(size * sigma_x), int(size * sigma_y)`, where `sigma_x`
        and `sigma_y` are the standard deviations of the distributions along the two axis. The kernel is normalized.

        Parameters
        ----------
        sigma_x:
            The standard deviation of a 2D gaussian distribution with axis symmetry, along the x axis.
        sigma_y:
            The standard deviation of a 2D gaussian distribution with axis symmetry, along the y axis.
        size:
            The multiplier to obtain the size of the kernel. In this case it corresponds to `size` sigmas.
            Default is `2.5`.
        """
        super(AsymmGaussKernel, self).__init__('asymm_gauss', sigma_x, sigma_y, size)

    def _generate_kernel(self):
        x_axis = np.arange(self.size_x) - self.size_x // 2
        y_axis = np.arange(self.size_y) - self.size_y // 2

        xx, yy = np.meshgrid(x_axis, y_axis, sparse=True)

        return np.exp(-0.5 * (xx ** 2 + yy ** 2)) / (2 * np.pi * self.width_x * self.width_y)
