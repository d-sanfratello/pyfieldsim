import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as scipysig
import warnings

from fieldsim.skysource import SkySource
from fieldsim.utils import DataType
from fieldsim.utils import ImageStatus

from fieldsim.psf import Kernel

from fieldsim.excep import WrongShapeError
from fieldsim.excep import NotInitializedError
from fieldsim.excep import ArgumentError
from fieldsim.excep import FieldNotInitializedError
from fieldsim.excep import UnexpectedDatatypeError
from fieldsim.warn import FieldNotInitializedWarning
from fieldsim.warn import FieldAlreadyInitializedWarning
from fieldsim.warn import LowLuminosityWarning


class Field:
    __valid_fields = ['true', 'ph_noise', 'background', 'psf', 'exposure', 'gain_map', 'dark_current']
    __attributes = {'true': 'true_field',
                    'ph_noise': 'w_ph_noise_field',
                    'background': 'w_background_field',
                    'psf': 'w_psf_field',
                    'exposure': 'recorded_field',
                    'gain_map': 'gain_map',
                    'dark_current': 'dark_current'}

    def __init__(self, shape):
        if not isinstance(shape, tuple) or len(shape) != 2:
            raise WrongShapeError

        self.shape = shape
        self.__initialized = False
        self.__ph_noise = False
        self.__background = False
        self.__psf = False
        self.__has_gain_map = False
        self.__has_dark_current = False
        self.__has_record = False

        self.max_signal_coords = None
        self.gain_map = None
        self.__mean_gain = None
        self.dark_current = None
        self.true_field = None
        self.w_ph_noise_field = None
        self.w_background_field = None
        self.w_psf_field = None
        self.recorded_field = None
        self.sources = None

        self.__work_field = None
        self.__pad = [self.shape[0] // 4, self.shape[1] // 4]

        self.__work_w_ph_noise = None
        self.__work_w_background = None
        self.__work_w_psf = None

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

        if self.__initialized and not force:
            warnings.warn("Field already initialized, use `force=True` argument to force re-initialization.",
                          FieldAlreadyInitializedWarning)
        elif not self.__initialized or force:
            rng = np.random.default_rng()

            work_shape = (self.shape[0] + 2 * self.__pad[0], self.shape[1] + 2 * self.__pad[1])
            self.__work_field = np.zeros(work_shape)

            # rand_distribution = rng.random(self.shape)
            rand_distribution = rng.random(self.__work_field.shape)
            stars_coords_arr = np.argwhere(rand_distribution <= density)

            self.sources = np.array([SkySource(coords).initialize(e_imf, e_lm, cst_lm) for coords in stars_coords_arr])
            self.true_field = np.zeros(self.shape)

            self.datatype = getattr(DataType(), datatype.upper())

            if self.datatype == DataType().LUMINOSITY:
                for source in self.sources:
                    # self.true_field[source.coords[0], source.coords[1]] = source.luminosity
                    self.__work_field[source.coords[0], source.coords[1]] = source.luminosity
            elif self.datatype == DataType().MAGNITUDE:
                for source in self.sources:
                    # self.true_field[source.coords[0], source.coords[1]] = source.magnitude
                    self.__work_field[source.coords[0], source.coords[1]] = source.magnitude
            elif self.datatype == DataType().MASS:
                for source in self.sources:
                    # self.true_field[source.coords[0], source.coords[1]] = source.mass
                    self.__work_field[source.coords[0], source.coords[1]] = source.mass
            self.true_field = self.__work_field[self.__pad[0]:-self.__pad[0], self.__pad[1]:-self.__pad[1]]
            self.max_signal_coords = np.unravel_index(np.argmax(self.true_field), self.shape)

            for source in self.sources:
                source.coords[0] -= self.__pad[0]
                source.coords[1] -= self.__pad[1]

            src_copy = filter(lambda x: (0 <= x.coords[0] < self.shape[0] and 0 <= x.coords[1] < self.shape[1]),
                              self.sources)

            self.sources = src_copy

            self.status = ImageStatus().SINGLESTARS
            self.__initialized = True

    def add_photon_noise(self, fluct='poisson', delta_time=1, force=False, multiply=False):
        if not isinstance(fluct, (str, float, int)):
            raise TypeError('`fluct` argument must be either a string or a number.')
        if isinstance(fluct, str) and fluct not in ['poisson']:
            raise ArgumentError

        if not isinstance(delta_time, (int, float)):
            raise TypeError('`delta_time` argument must be a number.')
        elif delta_time <= 0:
            raise ValueError('`delta_time` argument must be positive.')

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
            rng = np.random.default_rng()

            # exposed_true_field = self.true_field * delta_time
            exposed_true_field = self.__work_field * delta_time

            self.__work_w_ph_noise = np.zeros(self.__work_field.shape)

            if exposed_true_field[np.nonzero(exposed_true_field)].min() <= 1 and multiply:
                warnings.warn("Field is being multiplied by a constant so that the lowest luminosity star is at least \
                 \nabove 1 before adding photon noise fluctuations.",
                              LowLuminosityWarning)
                min_luminosity = exposed_true_field[np.nonzero(exposed_true_field)].min()
                min_exponent = - np.floor(np.log10(min_luminosity))
                # self.w_ph_noise_field = exposed_true_field * 10 ** min_exponent
                # self.w_ph_noise_field = np.round(self.w_ph_noise_field)
                # self.w_ph_noise_field = np.where(self.w_ph_noise_field > 0,
                #                                  rng.poisson(self.w_ph_noise_field), 0)
                self.__work_w_ph_noise = exposed_true_field * 10 ** min_exponent
                self.__work_w_ph_noise = np.round(self.__work_w_ph_noise)
                self.__work_w_ph_noise = np.where(self.__work_w_ph_noise > 0, rng.poisson(self.__work_w_ph_noise), 0)
            else:
                # self.w_ph_noise_field = np.round(exposed_true_field)
                # self.w_ph_noise_field = np.where(self.w_ph_noise_field > 0,
                #                                  rng.poisson(self.w_ph_noise_field), 0)
                self.__work_w_ph_noise = np.round(exposed_true_field)
                self.__work_w_ph_noise = np.where(self.__work_w_ph_noise > 0, rng.poisson(self.__work_w_ph_noise), 0)

            # self.w_ph_noise_field = np.where(self.w_ph_noise_field < 0, 0, self.w_ph_noise_field)
            self.__work_w_ph_noise = np.where(self.__work_w_ph_noise < 0, 0, self.__work_w_ph_noise)

            self.w_ph_noise_field = self.__work_w_ph_noise[self.__pad[0]:-self.__pad[0], self.__pad[1]:-self.__pad[1]]
            self.status = ImageStatus().PH_NOISE
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
            rng = np.random.default_rng()

            self.__work_w_background = np.zeros(self.__work_field.shape)

            if self.__ph_noise:
                loc = self.w_ph_noise_field[self.max_signal_coords]
                scale = rel_var * loc

                # self.w_background_field = self.w_ph_noise_field + rng.normal(loc, scale, self.shape)
                self.__work_w_background = self.__work_w_ph_noise + rng.normal(loc, scale, self.__work_w_ph_noise.shape)
            else:
                loc = self.true_field[self.max_signal_coords]
                scale = rel_var * loc

                # self.w_background_field = self.true_field + rng.normal(loc, scale, self.shape)
                self.__work_w_background = self.__work_field + rng.normal(loc, scale, self.__work_field.shape)

            # self.w_background_field = np.where(self.w_background_field < 0, 0, self.w_background_field)
            self.__work_w_background = np.where(self.__work_w_background < 0, 0, self.__work_w_background)

            self.w_background_field = self.__work_w_background[self.__pad[0]:-self.__pad[0],
                                      self.__pad[1]:-self.__pad[1]]
            self.status = ImageStatus().BACKGROUND
            self.__background = True

    def apply_psf(self, kernel, force=False):
        if not isinstance(kernel, Kernel):
            raise TypeError('`kernel` argument must be an instance of `fieldsim.psf.Kernel` class.')

        if not self.__initialized:
            raise FieldNotInitializedError

        if self.__psf and not force:
            warnings.warn("Field has already been convolved with a psf, use `force=True` argument to force psf again.",
                          FieldAlreadyInitializedWarning)
        elif not self.__psf or force:
            self.__work_w_psf = np.zeros(self.__work_field.shape)

            if not self.__ph_noise and not self.__background:
                # self.w_psf_field = scipysig.convolve2d(self.true_field, kernel.kernel, mode='same')
                self.__work_w_psf = scipysig.convolve2d(self.__work_field, kernel.kernel, mode='same')
            elif self.__ph_noise and not self.__background:
                # self.w_psf_field = scipysig.convolve2d(self.w_ph_noise_field, kernel.kernel, mode='same')
                self.__work_w_psf = scipysig.convolve2d(self.__work_w_ph_noise, kernel.kernel, mode='same')
            elif self.__background:
                # self.w_psf_field = scipysig.convolve2d(self.w_background_field, kernel.kernel, mode='same')
                self.__work_w_psf = scipysig.convolve2d(self.__work_w_background, kernel.kernel, mode='same')

            # self.w_psf_field = np.where(self.w_psf_field < 0, 0, self.w_psf_field)
            self.__work_w_psf = np.where(self.__work_w_psf < 0, 0, self.__work_w_psf)

            self.w_psf_field = self.__work_w_psf[self.__pad[0]:-self.__pad[0], self.__pad[1]:-self.__pad[1]]
            self.status = ImageStatus().PSF
            self.__psf = True

    def create_gain_map(self, mean_gain=1, rel_var=0.01, force=False):
        if not isinstance(mean_gain, (int, float)):
            raise TypeError('`mean_gain` must be a number.')
        if not isinstance(rel_var, (int, float)):
            raise TypeError('`rel_var` must be a number.')
        if not isinstance(force, bool):
            raise TypeError('`force` argument must be a bool.')

        if self.__has_gain_map and not force:
            warnings.warn("A gain map already exists, use `force=True` argument to force gain map generation again.",
                          FieldAlreadyInitializedWarning)
        elif not self.__has_gain_map or force:
            rng = np.random.default_rng()
            self.gain_map = rng.normal(mean_gain, rel_var * mean_gain, self.shape)

            self.gain_map = np.where(self.gain_map < 0, 0, self.gain_map)
            self.__mean_gain = mean_gain
            self.__has_gain_map = True

    def create_dark_current(self, b_fraction=0.1, rel_var=0.01, dk_c=1, force=False):
        if not isinstance(b_fraction, (int, float)):
            raise TypeError('`b_fraction` must be a number.')
        if not isinstance(rel_var, (int, float)):
            raise TypeError('`rel_var` must be a number.')
        if not isinstance(dk_c, (int, float)):
            raise TypeError('`dk_c` must be a number.')
        if not isinstance(force, bool):
            raise TypeError('`force` argument must be a bool.')

        if self.__has_dark_current and not force:
            warnings.warn("Dark current has already been generated, use `force=True` argument to force dark current \
             \ngeneration again.", FieldAlreadyInitializedWarning)
        elif not self.__has_dark_current or force:
            rng = np.random.default_rng()

            if self.__background:
                dark_current_mean = b_fraction * self.__mean_gain
            else:
                dark_current_mean = dk_c

            self.dark_current = rng.normal(dark_current_mean, rel_var * dark_current_mean, self.shape)
            self.__has_dark_current = True

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

    def record_field(self, kernel,
                     delta_time=1, ph_noise_fluct='poisson',
                     background_fluct='gauss', snr=10, bgnd_rel_var=0.05,
                     gain_mean=1, gain_rel_var=0.01,
                     dk_c_fraction=0.1, dk_c_rel_var=0.01, dk_c=1,
                     force=False):
        if not isinstance(force, bool):
            raise TypeError('`force` argument must be a bool.')
        if self.status == ImageStatus().NOTINIT:
            raise NotInitializedError
        elif self.datatype != DataType().LUMINOSITY:
            raise UnexpectedDatatypeError

        if self.__has_record and not force:
            warnings.warn("Complete field has been already generated, use `force=True` argument to force new exposure.",
                          FieldAlreadyInitializedWarning)
        elif not self.__has_record or force:
            self.add_photon_noise(fluct=ph_noise_fluct, delta_time=delta_time, force=True, multiply=False)
            self.add_background(fluct=background_fluct, snr=snr, rel_var=bgnd_rel_var, force=True)
            self.apply_psf(kernel=kernel, force=True)
            self.create_gain_map(mean_gain=gain_mean, rel_var=gain_rel_var, force=True)
            self.create_dark_current(b_fraction=dk_c_fraction, rel_var=dk_c_rel_var, dk_c=dk_c, force=True)

            self.recorded_field = self.gain_map * self.w_psf_field + self.dark_current
            self.__has_record = True

    @property
    def shot(self):
        if self.status == ImageStatus().NOTINIT:
            raise NotInitializedError
        elif self.status == ImageStatus().SINGLESTARS:
            return self.true_field
        elif self.status == ImageStatus().PH_NOISE:
            return self.w_ph_noise_field
        elif self.status == ImageStatus().BACKGROUND:
            return self.w_background_field
        elif self.status == ImageStatus().PSF:
            return self.w_psf_field
        elif self.status == ImageStatus().EXPOSURE:
            return self.recorded_field
