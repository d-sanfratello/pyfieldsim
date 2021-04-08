import matplotlib.pyplot as plt
import numpy as np
import warnings

from scipy.stats import norm

from fieldsim.excep import WrongShapeError
from fieldsim.warn import FieldNotInitializedWarning
from fieldsim.warn import FieldAlreadyInitializedWarning

from fieldsim.skysource import SkySource


class Field:
    __valid_fields = ['true']
    __attributes = {'true': 'true_field'}

    def __init__(self, shape):
        if not isinstance(shape, tuple) or len(shape) != 2:
            raise WrongShapeError

        self.shape = shape
        self.__initialized = False

        self.true_field = None
        self.sources = None

    def initialize_field(self, density=0.05, force=False,
                         e_imf=2.4, e_lm=3, cst_lm=1):
        if self.__initialized and not force:
            warnings.warn("Field already initialized, use `force=True` argument to force re-initialization.",
                          FieldAlreadyInitializedWarning)

        self.__initialized = True

        rand_distribution = np.random.random(self.shape)
        stars_coords_array = np.argwhere(rand_distribution <= density)

        self.sources = np.array([SkySource(coords).initialize(e_imf, e_lm, cst_lm) for coords in stars_coords_array])
        self.true_field = np.zeros(self.shape)

        for source in self.sources:
            self.true_field[source.coords[0], source.coords[1]] = source.luminosity

    def show_field(self, field='true'):
        if not isinstance(field, str):
            raise TypeError('`field` argument must be a string.')
        elif isinstance(field, str) and field not in self.__valid_fields:
            raise ValueError('`field` argument not in accepted list of valid fields.')

        image = getattr(self, self.__attributes[field])

        fig = plt.figure()
        ax1 = fig.gca()
        ax1.grid()

        im = ax1.imshow(image, origin='lower', cmap='binary')
        ax1.set_xlabel('RA')
        ax1.set_ylabel('DEC')

        cbar = fig.colorbar(im)
        cbar.ax.set_ylabel('L$_X$')
        plt.show()
