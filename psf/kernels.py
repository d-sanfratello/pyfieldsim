import numpy as np

from fieldsim.psf import Kernel


class GaussKernel(Kernel):
    def __init__(self, sigma, size=2.5):
        super(GaussKernel, self).__init__('gauss', sigma, sigma, size)

    def _generate_kernel(self):
        x_axis = np.arange(self.size_x) - self.size_x // 2
        y_axis = np.arange(self.size_y) - self.size_y // 2

        xx, yy = np.meshgrid(x_axis, y_axis, sparse=True)

        return np.exp(-0.5 * (xx**2 + yy**2)) / (2 * np.pi * self.width_x * self.width_y)


class AsymmGaussKernel(Kernel):
    def __init__(self, sigma_x, sigma_y, size=2.5):
        super(AsymmGaussKernel, self).__init__('asymm_gauss', sigma_x, sigma_y, size)

    def _generate_kernel(self):
        x_axis = np.arange(self.size_x) - self.size_x // 2
        y_axis = np.arange(self.size_y) - self.size_y // 2

        xx, yy = np.meshgrid(x_axis, y_axis, sparse=True)

        return np.exp(-0.5 * (xx ** 2 + yy ** 2)) / (2 * np.pi * self.width_x * self.width_y)
