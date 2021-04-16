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
        self.__background = False
        self.__has_gain_map = False

        self.max_signal_coords = None
        self.gain_map = None
        self.true_field = None
        self.ph_noise_field = None
        self.background_field = None
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
                self.max_signal_coords = np.unravel_index(np.argmax(self.true_field), self.shape)
            elif self.datatype == DataType().MAGNITUDE:
                for source in self.sources:
                    self.true_field[source.coords[0], source.coords[1]] = source.magnitude
            elif self.datatype == DataType().MASS:
                for source in self.sources:
                    self.true_field[source.coords[0], source.coords[1]] = source.mass
        elif self.__initialized and not force:
            warnings.warn("Field already initialized, use `force=True` argument to force re-initialization.",
                          FieldAlreadyInitializedWarning)

    def add_photon_noise(self, fluct='poisson', force=False, multiply=True):
        if not isinstance(fluct, (str, float, int)):
            raise TypeError('`fluct` argument must be either a string or a number.')
        if isinstance(fluct, str) and fluct not in ['poisson']:
            raise ArgumentError
        if not isinstance(force, bool):
            raise TypeError('`force` argument must be a bool.')
        if not isinstance(multiply, bool):
            raise TypeError('`multiply` argument must be a bool.')
        if not self.__initialized:
            raise FieldNotInitializedError
        if self.datatype != DataType().LUMINOSITY:
            raise UnexpectedDatatypeError

        if self.__ph_noise and not force:
            warnings.warn("Field already has photon noise, use `force=True` argument to force photon noise again.",
                          FieldAlreadyInitializedWarning)
        elif not self.__ph_noise or force:
            self.status = ImageStatus().PH_NOISE

            if self.true_field[np.nonzero(self.true_field)].min() <= 1 and multiply:
                warnings.warn("Field is being multiplied by a constant so that the lowest luminosity star is at least \
                 \nabove 1 before adding photon noise fluctuations.",
                              LowLuminosityWarning)
                min_luminosity = self.true_field[np.nonzero(self.true_field)].min()
                min_exponent = - np.floor(np.log10(min_luminosity))
                self.ph_noise_field = self.true_field * 10**min_exponent
                self.ph_noise_field = np.round(self.ph_noise_field)
                self.ph_noise_field = np.where(self.ph_noise_field > 0,
                                               np.random.Generator.poisson(self.ph_noise_field), 0)
            else:
                self.ph_noise_field = np.round(self.true_field)
                self.ph_noise_field = np.where(self.ph_noise_field > 0,
                                               np.random.Generator.poisson(self.ph_noise_field), 0)

            self.ph_noise_field = np.where(self.ph_noise_field < 0, 0, self.ph_noise_field)
            self.__ph_noise = True

    def add_background(self, fluct='gauss', snr=10, rel_var=0.05, force=False):
        if not isinstance(fluct, (str, float, int)):
            raise TypeError('`fluct` argument must be either a string or a number.')
        if isinstance(fluct, str) and fluct not in ['gauss']:
            raise ArgumentError

        if not isinstance(snr, (int, float)):
            raise TypeError('`snr` argument must be a number.')
        elif snr <= 0:
            raise ValueError('`snr` argument must be positive.')

        if not isinstance(rel_var, (int, float)):
            raise TypeError('`rel_var` argument must be a number.')
        elif rel_var < 0 or rel_var > 1:
            raise ValueError('`snr` argument must be positive.')

        if not isinstance(force, bool):
            raise TypeError('`force` argument must be a bool.')
        if not self.__initialized:
            raise FieldNotInitializedError
        if self.datatype != DataType().LUMINOSITY:
            raise UnexpectedDatatypeError

        if self.__background and not force:
            warnings.warn("Field already has background, use `force=True` argument to force background again.",
                          FieldAlreadyInitializedWarning)
        elif not self.__background or force:
            self.status = ImageStatus().BACKGROUND

            if self.__ph_noise:
                loc = self.ph_noise_field[self.max_signal_coords]
                scale = rel_var * loc

                self.background_field = self.ph_noise_field + np.random.Generator.normal(loc, scale, self.shape)
            else:
                loc = self.true_field[self.max_signal_coords]
                scale = rel_var * loc

                self.background_field = self.true_field + np.random.Generator.normal(loc, scale, self.shape)

            self.background_field = np.where(self.background_field < 0, 0, self.background_field)
            self.__background = True

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
        elif self.status == ImageStatus().PH_NOISE:
            return self.ph_noise_field
        elif self.status == ImageStatus().BACKGROUND:
            return self.background_field
