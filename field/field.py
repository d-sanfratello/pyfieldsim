import matplotlib.pyplot as plt
import numpy as np
import warnings

from scipy.stats import norm

from fieldsim.skysource import SkySource
from fieldsim.utils import DataType
from fieldsim.utils import ImageStatus

from fieldsim.excep import WrongShapeError
from fieldsim.excep import NotInitializedError
from fieldsim.excep import ArgumentError
from fieldsim.excep import FieldNotInitializedError
from fieldsim.excep import UnexpectedDatatypeError
from fieldsim.warn import FieldNotInitializedWarning
from fieldsim.warn import FieldAlreadyInitializedWarning
from fieldsim.warn import LowLuminosityWarning


class Field:
    __valid_fields = ['true', 'ph_noise', 'gain_map']
    __attributes = {'true': 'true_field',
                    'ph_noise': 'ph_noise_field',
                    'gain_map': 'gain_map'}

    def __init__(self, shape):
        if not isinstance(shape, tuple) or len(shape) != 2:
            raise WrongShapeError

        self.shape = shape
        self.__initialized = False
        self.__ph_noise = False
        self.__has_gain_map = False

        self.gain_map = None
        self.true_field = None
        self.ph_noise_field = None
        self.sources = None

        self.status = ImageStatus().NOTINIT
        self.datatype = DataType().NOTINIT

    def initialize_field(self, density=0.05, force=False,
                         e_imf=2.4, e_lm=3, cst_lm=1,
                         datatype='luminosity'):
        if not isinstance(force, bool):
            raise TypeError("`force` argument must be a bool.")

        if not isinstance(datatype, str):
            raise TypeError("`datatype` argument must be a string.")
        elif datatype not in ['luminosity', 'magnitude', 'mass']:
            raise ValueError("`datatype` argument must be \"luminosity\", \"magnitude\" or \"mass\".")

        if not self.__initialized or force:
            self.__initialized = True

            rand_distribution = np.random.random(self.shape)
            stars_coords_array = np.argwhere(rand_distribution <= density)

            self.sources = np.array([SkySource(coords).initialize(e_imf, e_lm, cst_lm) for coords in stars_coords_array])
            self.true_field = np.zeros(self.shape)

            self.status = ImageStatus().SINGLESTARS
            self.datatype = getattr(DataType(), datatype.upper())

            if self.datatype == DataType().LUMINOSITY:
                for source in self.sources:
                    self.true_field[source.coords[0], source.coords[1]] = source.luminosity
            elif self.datatype == DataType().MAGNITUDE:
                for source in self.sources:
                    self.true_field[source.coords[0], source.coords[1]] = source.magnitude
            elif self.datatype == DataType().MASS:
                for source in self.sources:
                    self.true_field[source.coords[0], source.coords[1]] = source.mass
        elif self.__initialized and not force:
            warnings.warn("Field already initialized, use `force=True` argument to force re-initialization.",
                          FieldAlreadyInitializedWarning)

    def add_photon_noise(self, fluct='poisson', force=False):
        if not isinstance(fluct, (str, float, int)):
            raise TypeError('`fluct` argument must be either a string or a number.')
        if isinstance(fluct, str) and not fluct in ['poisson']:
            raise ArgumentError
        if not isinstance(force, bool):
            raise TypeError('`force` argument must be a bool.')
        if not self.__initialized:
            raise FieldNotInitializedError
        if self.datatype != DataType().LUMINOSITY:
            raise UnexpectedDatatypeError

        if self.__ph_noise and not force:
            warnings.warn("Field already has photon noise, use `force=True` argument to force photon noise again.",
                          FieldAlreadyInitializedWarning)
        elif not self.__ph_noise or force:
            if self.true_field.max() <= 1:
                warnings.warn("Luminosity is too low to discretize photons. Field is being multiplied by 100.",
                              LowLuminosityWarning)
                self.ph_noise_field = 100 * self.true_field
                self.ph_noise_field = np.round(self.ph_noise_field)
                self.ph_noise_field = np.where(self.ph_noise_field > 0,
                                               np.random.poisson(self.ph_noise_field), 0)
            else:
                self.ph_noise_field = np.round(self.true_field)
                self.ph_noise_field = np.where(self.ph_noise_field > 0,
                                               np.random.poisson(self.ph_noise_field), 0)

            self.ph_noise_field = np.where(self.ph_noise_field < 0,
                                           0, self.ph_noise_field)
            self.__ph_noise = True

    def create_gain_map(self, mean_gain=1, rel_var=0.01, force=False):
        if not isinstance(mean_gain, (int, float)):
            raise TypeError('`mean_gain` must be a number.')
        if not isinstance(rel_var, (int, float)):
            raise TypeError('`rel_var` must be a number.')
        if not isinstance(force, bool):
            raise TypeError('`force` argument must be a bool.')

        if self.__has_gain_map and not force:
            warnings.warn("A gain map already exists")
        elif not self.__has_gain_map or force:
            self.gain_map = np.random.normal(mean_gain, rel_var * mean_gain, self.shape)

            self.gain_map = np.where(self.gain_map < 0, 0, self.gain_map)
            self.__has_gain_map = True

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
        cbar.ax.set_ylabel(self.datatype.capitalize())
        plt.show()

    @property
    def shot(self):
        if self.status == ImageStatus().NOTINIT:
            raise NotInitializedError
        elif self.status == ImageStatus().SINGLESTARS:
            return self.true_field
