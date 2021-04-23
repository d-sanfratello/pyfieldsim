import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as scipysig
import warnings

from ..skysource import SkySource
from ..utils import DataType
from ..utils import ImageStatus

from ..psf import Kernel

from ..excep import WrongShapeError
from ..excep import NotInitializedError
from ..excep import ArgumentError
from ..excep import UnexpectedDatatypeError
from ..warn import FieldAlreadyInitializedWarning
from ..warn import LowLuminosityWarning


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
        """
        Class that initializates a `Field` instance, to simulate an exposure through a telescope.

        At initialization, the class saves the shape of the output field. Every simulation step is described in its
        own method. See `dir(Field)` for the list of available methods.

        Parameters
        ----------
        shape: `tuple` of size 2
            Tuple representing the shape of the output field.

        Raises
        ------
        `WrongShapeError`:
            If `shape` is not a tuple.

        Examples
        --------
        >>> # Initializing a field of shape (100, 100).
        >>> fld = Field((100, 100))
        """
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
        self.__mean_background = None
        self.dark_current = None
        self.true_field = None
        self.w_ph_noise_field = None
        self.w_background_field = None
        self.w_psf_field = None
        self.recorded_field = None
        self.sources = None

        self.__aux_field = None
        self.__pad = [self.shape[0] // 4, self.shape[1] // 4]

        self.__aux_w_ph_noise = None
        self.__aux_w_background = None
        self.__aux_w_psf = None

        self.status = ImageStatus().NOTINIT
        self.datatype = DataType().NOTINIT

    def initialize_field(self, density=0.05, e_imf=2.4, e_lm=3, cst_lm=1, seed=None,
                         datatype='luminosity', force=False):
        """
        Method that initializates the field randomly generating the stars in it as instances of the
        `skysource.SkySource` class.

        It defines an auxiliary field `Field.__aux_field` whose shape is the shape defined at initialization with a
        padding added. This will be useful at later stages of simulations, where the convolution with the psf kernel
        would generate boundary artifacts. In this implementation, though, the artifacts are far from the actual
        boundary of the image, which is realized by cropping the auxiliary field at its center. The size of the padding
        is, in each direction, equal to 1/4 of the corresponding edge's shape.

        After initializing the random number generator from numpy, the auxiliary field is initialized with random
        numbers in the interval [0, 1). If the generated number in a given pixel is below a threshold given by
        `density`, a star is initialized at those coordinates, by using the `skysource.SkySource` class.

        The auxiliary field is, then, cropped to obtain the actual field on the CCD. Also, in the `Field.sources`
        attribute the stars belonging to the padding area are deleted. This is to simulate their existence (and
        effect on the field when psf are taken into account) but without being able to measure their distance from
        the exposed field.

        There are three possible settings for the field initialization as `datatype`: 'luminosity', 'mass' and
        'magnitude', where the last two are mainly for debugging reasongs. Every step of the simulation, indeed,
        require the `datatype` to be 'luminosity', as defined in `utils.DataType` class.

        Parameters
        ----------
        density: `number` in [0, 1]
            Expected number density of stars in the field. Default is `0.05`.
        e_imf: `number`
            Exponent of the IMF powerlaw, to be passed to the `skysource.SkySource.random_cimf` method. Default is
            `2.4`.
        e_lm: `number`
            Exponent of the mass-luminosity relation powerlaw, to be passed to the `skysource.SkySource.lm_relation`
            method. Default is `3`.
        cst_lm: `number`
            Factor in front of the mass-luminosity relation powerlaw, to be passed to the
            `skysource.SkySource.lm_relation` method. Default is `1`.
        seed: `int` or `None`
            If of type `int`, passes the seed to the pseudo-random number generator of numpy. If `None`,
            numpy generates a seed by itself. Default is `None`.
        datatype: `'luminosity'`, `'mass'` or `'magnitude'`
            Defines the datatype represented in the field. Default is `'luminosity'`.
        force: `bool`
            Flag to force re-initialization of the field if it had already been initialized. Default is `False`.

        Raises
        ------
        `TypeError`:
            If `datatype` is not a string or if `force` is not a `bool`.
        `ValueError`:
            If `datatype` is not `'luminosity'`, `'mass'` or `'magnitude'`.
        `FieldAlreadyInitializedWarning`:
            If the field has been previously initialized. Execution is not halted, but the field is not updated.

        Examples
        --------
        >>> # Initializes a field with a Salpeter IMF and a Main-Sequence L-M relation but with enhanced luminosity.
        >>> # The density of stars has been set to the 2%. The data that will be represented in the field is in units of
        >>> # luminosity.
        >>> fld.initialize_field(density=0.02, e_imf=2.4, e_lm=3, cst_lm=100, datatype='luminosity')

        See Also
        --------
        `skysource.SkySource`, `utils.DataType`.
        """
        if not isinstance(seed, int) and seed is not None:
            raise TypeError("`seed` argument must be either an `int` or `None`.")

        if not isinstance(datatype, str):
            raise TypeError("`datatype` argument must be a string.")
        elif datatype not in ['luminosity', 'magnitude', 'mass']:
            raise ValueError("`datatype` argument must be \"luminosity\", \"magnitude\" or \"mass\".")

        if not isinstance(force, bool):
            raise TypeError("`force` argument must be a bool.")

        if self.__initialized and not force:
            warnings.warn("Field already initialized, use `force=True` argument to force re-initialization.",
                          FieldAlreadyInitializedWarning)
        elif not self.__initialized or force:
            rng = np.random.default_rng(seed=seed)

            # Generation of the auxiliary field
            aux_shape = (self.shape[0] + 2 * self.__pad[0], self.shape[1] + 2 * self.__pad[1])
            self.__aux_field = np.zeros(aux_shape)

            # Random generation of the stars
            rand_distribution = rng.random(self.__aux_field.shape)
            stars_coords_arr = np.argwhere(rand_distribution <= density)

            # Saving the sources and randomly generating their masses
            self.sources = np.array([SkySource(coords).initialize(e_imf, e_lm, cst_lm, seed=seed) for coords in
                                     stars_coords_arr])
            self.true_field = np.zeros(self.shape)

            self.datatype = getattr(DataType(), datatype.upper())

            if self.datatype == DataType().LUMINOSITY:
                for source in self.sources:
                    self.__aux_field[source.coords[0], source.coords[1]] = source.luminosity
            elif self.datatype == DataType().MAGNITUDE:
                for source in self.sources:
                    self.__aux_field[source.coords[0], source.coords[1]] = source.magnitude
            elif self.datatype == DataType().MASS:
                for source in self.sources:
                    self.__aux_field[source.coords[0], source.coords[1]] = source.mass

            # Cropping of the auxiliary field, to obtain the CCD image
            self.true_field = self.__aux_field[self.__pad[0]:-self.__pad[0], self.__pad[1]:-self.__pad[1]]
            # Max signal in the cropped field is saved, to use it at background generation with the SNR
            self.max_signal_coords = np.unravel_index(np.argmax(self.true_field), self.shape)

            # Deleting the sources outside the CCD image from the list
            for source in self.sources:
                source.coords[0] -= self.__pad[0]
                source.coords[1] -= self.__pad[1]

            src_copy = filter(lambda x: (0 <= x.coords[0] < self.shape[0] and 0 <= x.coords[1] < self.shape[1]),
                              self.sources)

            self.sources = src_copy

            self.status = ImageStatus().SINGLESTARS
            self.__initialized = True

    def add_photon_noise(self, delta_time=1, force=False, multiply=False):
        """
        Method that applies photon fluctuations to the luminosity of stars in the field.

        The auxiliary field from the initialization is multiplied by `delta_time`, used to simulate longer (or shorter)
        exposures this is used to create another auxiliary field for the photon noise. These values are used,
        rounded to the closer integer, as the mean values of a random poisson number generator.

        Any negative number is, then, set to zero and the auxiliary field is cropped again and saved in the
        `w_ph_noise_field` attribute.

        Parameters
        ----------
        delta_time: `number` > 0
            Factor that represents a longer or shorter exposure. Default is `1`.
        force: `bool`
            Flag to force re-initialization of the field if it had already been initialized. Default is `False`.
        multiply: `bool`
            Flag to force every star in the field to have a luminosity value at least greater than 1. If `True`,
            the minimum exponent of the field is evaluated and the whole field is multiplied by this order of magnitude.

        Raises
        ------
        `TypeError`:
            If `delta_time` is not a number, or if `force` or `multiply` are not `bool`.
        `ValueError`:
            If `delta_time` is less or equal to 0.
        `NotInitializedError`:
            If the field has not been initialized, yet.
        `UnexpectedDatatypeError`:
            If the `self.datatype` attribute is not equal to `utils.DataType().LUMINOSITY`.
        `FieldAlreadyInitializedWarning`:
            If the field has been previously initialized. Execution is not halted, but the field is not updated.

        Examples
        --------
        >>> # Simulation of the photon noise for a field, supposing an exposition time 100 times longer than the
        >>> # default.
        >>> fld.add_photon_noise(delta_time=100)
        """
        if not isinstance(delta_time, (int, float)):
            raise TypeError('`delta_time` argument must be a number.')
        elif delta_time <= 0:
            raise ValueError('`delta_time` argument must be positive.')

        if not isinstance(force, bool):
            raise TypeError('`force` argument must be a bool.')
        if not isinstance(multiply, bool):
            raise TypeError('`multiply` argument must be a bool.')
        if not self.__initialized:
            raise NotInitializedError
        if self.datatype != DataType().LUMINOSITY:
            raise UnexpectedDatatypeError

        if self.__ph_noise and not force:
            warnings.warn("Field already has photon noise, use `force=True` argument to force photon noise again.",
                          FieldAlreadyInitializedWarning)
        elif not self.__ph_noise or force:
            rng = np.random.default_rng()

            # Generating the auxiliary fields and simulating exposure time variation with `delta_time`
            exposed_true_field = self.__aux_field * delta_time
            self.__aux_w_ph_noise = np.zeros(self.__aux_field.shape)

            if exposed_true_field[np.nonzero(exposed_true_field)].min() <= 1 and multiply:
                warnings.warn("Field is being multiplied by a constant so that the lowest luminosity star is at least \
                 \nabove 1 before adding photon noise fluctuations.",
                              LowLuminosityWarning)
                # Simulation if all luminosities have been set as >= 1.
                min_luminosity = exposed_true_field[np.nonzero(exposed_true_field)].min()
                min_exponent = - np.floor(np.log10(min_luminosity))

                self.__aux_w_ph_noise = exposed_true_field * 10 ** min_exponent

                # Poisson generation of photon noise arount the (integer) luminosity of the star as mean
                self.__aux_w_ph_noise = np.where(self.__aux_w_ph_noise > 0, rng.poisson(self.__aux_w_ph_noise), 0)
            else:
                # Simulation if all luminosities have been left as after exposure time simulation.
                self.__aux_w_ph_noise = exposed_true_field
                # Poisson generation of photon noise arount the (integer) luminosity of the star as mean
                self.__aux_w_ph_noise = np.where(self.__aux_w_ph_noise > 0, rng.poisson(self.__aux_w_ph_noise), 0)

            self.__aux_w_ph_noise = np.where(self.__aux_w_ph_noise < 0, 0, self.__aux_w_ph_noise)

            # Crop of the field
            self.w_ph_noise_field = self.__aux_w_ph_noise[self.__pad[0]:-self.__pad[0], self.__pad[1]:-self.__pad[1]]
            self.status = ImageStatus().PH_NOISE
            self.__ph_noise = True

    def add_background(self, fluct='gauss', snr=10, rel_var=0.05, force=False):
        """
        Method that generates a background for the field.

        This works both if the simulation of the photon noise (see `Field.add_photon_noise`) has been added or not.

        In case the photon noise has been generated, the background is generated starting from its corresponding
        auxiliary field, otherwise the base auxiliary field is used. This method looks for the brightest object in the
        field (whose location had been saved at initialization, see `Field.initialize_field`) and scaled down by the
        indicated Signal to Noise Ratio `snr`.

        In every pixel of an auxiliary field larger than the actual CCD field the noise is generated with a random
        gaussian number generator. The mean of the noise is set as described before, while the standard deviation is
        determined from its relative value `rel_var`.

        Any negative number is, then, set to zero and the auxiliary field is cropped again and saved in the
        `w_background_field` attribute.

        Parameters
        ----------
        fluct: `str`
            Distribution that the background follows. At the moment is a placeholder, since it is always set as a
            gaussian. Default is `'gaussian'`.
        snr: `number` > 0
            Estimate of the Signal to Noise Ratio, used to generate the background. Default is `10`.
        rel_var: `number` in [0, 1]
            Relative dispersion of the background distribution. It is used to determine the standard deviation of a
            gaussian background. Default is `0.05`.
        force: `bool`
            Flag to force re-initialization of the field if it had already been initialized. Default is `False`.

        Raises
        ------
        `TypeError`:
            If `fluct` is not a string, if `snr` or `rel_var` are not `numbers` or if `force` is not `bool`.
        `ArgumentError`:
            If `fluct` is not `'gauss'`. This is the only allowed value, at the moment.
        `NotInitializedError`:
            If the field has not been initialized, yet.
        `UnexpectedDatatypeError`:
            If the `self.datatype` attribute is not equal to `utils.DataType().LUMINOSITY`.
        `FieldAlreadyInitializedWarning`:
            If the field has been previously initialized. Execution is not halted, but the field is not updated.

        Examples
        --------
        >>> # Simulation of a low signal field (SNR of brightes star is 5) and a larger standard deviation from the
        >>> # mean: a relative dispersion of 10% from the mean of the background.
        >>> fld.add_background(snr=5, rel_var=0.1)

        See Also
        --------
        `Field.initialize_field` and `Field.add_photon_noise`.
        """
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
            raise NotInitializedError
        if self.datatype != DataType().LUMINOSITY:
            raise UnexpectedDatatypeError

        if self.__background and not force:
            warnings.warn("Field already has background, use `force=True` argument to force background again.",
                          FieldAlreadyInitializedWarning)
        elif not self.__background or force:
            rng = np.random.default_rng()

            self.__aux_w_background = np.zeros(self.__aux_field.shape)

            if self.__ph_noise:
                # Generation of background with a normal distribution of mean given by the max value in the field
                # divided by the SNR.
                loc = self.w_ph_noise_field[self.max_signal_coords] / snr
                scale = rel_var * loc

                self.__aux_w_background = self.__aux_w_ph_noise + rng.normal(loc, scale, self.__aux_w_ph_noise.shape)
            else:
                # Generation of background with a normal distribution of mean given by the max value in the field
                # divided by the SNR.
                loc = self.true_field[self.max_signal_coords] / snr
                scale = rel_var * loc

                self.__aux_w_background = self.__aux_field + rng.normal(loc, scale, self.__aux_field.shape)

            self.__aux_w_background = np.where(self.__aux_w_background < 0, 0, self.__aux_w_background)

            # Crop of the auxiliary field.
            self.w_background_field = self.__aux_w_background[self.__pad[0]:-self.__pad[0],
                                                              self.__pad[1]:-self.__pad[1]]

            self.__mean_background = loc
            self.status = ImageStatus().BACKGROUND
            self.__background = True

    def apply_psf(self, kernel, force=False):
        """
        Method that applies a Point Spread Function (psf) to the field.

        This method creates an auxiliary field larger than the CCD's field (see also the `Field.initialize_field`
        method) from the `w_background_field` attribute, if available. If not, it first looks for the `w_ph_noise_field`
        attribute and, eventually, applies the psf to the base field.

        Convolution is made with the `scipy.signal.convolve2d` function, setting the mode as `'same'`. The
        operation is done over the selected auxiliary field and a kernel from `psf.kernels` kernel or a user-defined
        `psf.Kernel` object. This creates a field that has the same dimensions of the larger between the auxiliary field
        and the kernel, but has some artifacts due to boundary effects. To avoid artifacts due to zero condition
        boundaries or to interpolation, all the simulations are done on a way larger area that will, indeed, have
        artifacts, but the actual field is obtained after cropping the auxiliary field. This also allows for a further
        level of simulation, in which sources (or background) which is not on the CCD can actually have an effect over
        the recorded exposure.

        Any negative number is, then, set to zero and the auxiliary field is cropped again and saved in the
        `w_psf_field` attribute.

        Parameters
        ----------
        kernel: `fieldsim.psf.Kernel`
            Kernel of the point spread function. See `fieldsim.psf.Kernel`.
        force: `bool`
            Flag to force re-initialization of the field if it had already been initialized. Default is `False`.

        Raises
        ------
        `TypeError`:
            If `kernel` is not an instance of `psf.Kernel`.
        `NotInitializedError`:
            If the field has not been initialized, yet.
        `FieldAlreadyInitializedWarning`:
            If the field has been previously initialized. Execution is not halted, but the field is not updated.

        Examples
        --------
        >>> # Import of kernels. The gaussian one is used to initialize the psf.
        >>> from fieldsim.psf.kernels import GaussKernel
        >>> psf = GaussKernel(sigma=3, size=2.5)
        >>> # The psf is applied to the field.
        >>> fld.apply_psf(psf)

        See Also
        --------
        `psf.Kernel`, `Field.initialize_field` and `Field.add_photon_noise`. See also `scipy.signal.convolve2d`
        documentation.
        """
        if not isinstance(kernel, Kernel):
            raise TypeError('`kernel` argument must be an instance of `fieldsim.psf.Kernel` class.')

        if not self.__initialized:
            raise NotInitializedError

        if self.__psf and not force:
            warnings.warn("Field has already been convolved with a psf, use `force=True` argument to force psf again.",
                          FieldAlreadyInitializedWarning)
        elif not self.__psf or force:
            self.__aux_w_psf = np.zeros(self.__aux_field.shape)

            # Applying the psf kernel convolution on the auxiliary field corresponding on the actual datatyoe of the
            # simulation.
            if not self.__ph_noise and not self.__background:
                self.__aux_w_psf = scipysig.convolve2d(self.__aux_field, kernel.kernel, mode='same')
            elif self.__ph_noise and not self.__background:
                self.__aux_w_psf = scipysig.convolve2d(self.__aux_w_ph_noise, kernel.kernel, mode='same')
            elif self.__background:
                self.__aux_w_psf = scipysig.convolve2d(self.__aux_w_background, kernel.kernel, mode='same')

            self.__aux_w_psf = np.where(self.__aux_w_psf < 0, 0, self.__aux_w_psf)

            # Crop of the field.
            self.w_psf_field = self.__aux_w_psf[self.__pad[0]:-self.__pad[0], self.__pad[1]:-self.__pad[1]]
            self.status = ImageStatus().PSF
            self.__psf = True

    def create_gain_map(self, mean_gain=1, rel_var=0.01, force=False):
        """
        Method that creates a gain map for the simulated CCD.

        This method creates a gain map for the CCD and stores it into the `Field.gain_map` attribute. Simulation is
        done with a random gaussian number generator, frum numpy, with mean set from `mean_gain` and `rel_var` used
        to derive the standard deviation of the distribution.

        Ideally, this method should be called only once, since it simulates the differences in the production of each
        pixel, that do not fluctuate from observation to observation.

        Any pixel whose gain is negative, has gain set to zero.

        Parameters
        ----------
        mean_gain: `number` >= 0
            Mean value of the gain of the CCD. Default is 1.
        rel_var: `number` >= 0
            Relative dispersion of the gain of the CCD. Default is 0.01.
        force: `bool`
            Flag to force re-initialization of the field if it had already been initialized. Default is `False`.

        Raises
        ------
        `TypeError`:
            If `mean_gain` or `rel_var` are not numbers or if `force` is not a `bool`.
        `ValueError`:
            If `mean_gain` or `rel_var` are negative.
        `FieldAlreadyInitializedWarning`:
            If the gain map had been previously initialized. Execution is not halted, but the gain map is not updated.

        Examples
        --------
        >>> # This CCD has a mean gain of 1, with a dispersion of 1%.
        >>> fld.create_gain_map(mean_gain=1, rel_var=0.01)
        """
        if not isinstance(mean_gain, (int, float)):
            raise TypeError('`mean_gain` must be a number.')
        elif mean_gain < 0:
            raise ValueError('`mean_gain` cannot be less than zero.')

        if not isinstance(rel_var, (int, float)):
            raise TypeError('`rel_var` must be a number.')
        elif rel_var < 0:
            raise ValueError('`rel_var` cannot be less than zero.')

        if not isinstance(force, bool):
            raise TypeError('`force` argument must be a bool.')

        if self.__has_gain_map and not force:
            warnings.warn("A gain map already exists, use `force=True` argument to force gain map generation again.",
                          FieldAlreadyInitializedWarning)
        elif not self.__has_gain_map or force:
            rng = np.random.default_rng()

            # Creation of the gain map of the CCD as a normal distribution centered on `mean_gain`.
            self.gain_map = rng.normal(mean_gain, rel_var * mean_gain, self.shape)

            self.gain_map = np.where(self.gain_map < 0, 0, self.gain_map)
            self.__mean_gain = mean_gain
            self.__has_gain_map = True

    def create_dark_current(self, b_fraction=0.1, rel_var=0.01, dk_c=1, force=False):
        """
        Method that simulates a dark current inside the CCD.

        This method works either after having simulated a background or by manual injection of the mean value. If a
        background has been generated, this method reads a private attribute containing the mean value of the
        background and determines the mean of the dark current by calculating:

            dark_current_mean = b_fraction * Field.__mean_background

        If, on the other hand, the background isn't available, the dark current mean value is set from the `dk_c`
        argument.

        Parameters
        ----------
        b_fraction: `number` >= 0
            Fraction of the mean background to be set as mean for the distribution of the dark current of the CCD.
            Default is 0.1 (optional).
        rel_var: `number >= 0
            Relative dispersion of the dark current of the CCD. Default is 0.01.
        dk_c: `number` >= 0
            Mean value of the dark current of the CCD, if no background is available. Default is 1 (optional).
        force: `bool`
            Flag to force re-initialization of the field if it had already been initialized. Default is `False`.

        Raises
        ------
        `TypeError`:
            If `b_fraction`, `rel_var` or `dk_c` are not numbers or if `force` is not a `bool`.
        `ValueError`:
            If `b_fraction`, `rel_var` or `dk_c` are negative.
        `FieldAlreadyInitializedWarning`:
            If the dark current had been previously simulated. Execution is not halted, but the dark current is not
            updated.

        Examples
        --------
        >>> # This call simulates a dark current with a mean value equal to 10% of the mean background and a relative
        >>> # dispersion of 5%.
        >>> fld.create_dark_current(b_fraction=0.1, rel_var=5e-3)

        See Also
        --------
        `Field.add_background`.
        """
        if not isinstance(b_fraction, (int, float)):
            raise TypeError('`b_fraction` must be a number.')
        elif b_fraction < 0:
            raise ValueError('`b_fraction` cannot be less than zero.')

        if not isinstance(rel_var, (int, float)):
            raise TypeError('`rel_var` must be a number.')
        elif rel_var < 0:
            raise ValueError('`rel_var` cannot be less than zero.')

        if not isinstance(dk_c, (int, float)):
            raise TypeError('`dk_c` must be a number.')
        elif dk_c < 0:
            raise ValueError('`dk_c` cannot be less than zero.')

        if not isinstance(force, bool):
            raise TypeError('`force` argument must be a bool.')

        if self.__has_dark_current and not force:
            warnings.warn("Dark current has already been generated, use `force=True` argument to force dark current \
             \ngeneration again.", FieldAlreadyInitializedWarning)
        elif not self.__has_dark_current or force:
            rng = np.random.default_rng()

            if self.__background:
                # If the background has been simulated, the dark current's mean is a fraction of the used mean
                # background.
                dark_current_mean = b_fraction * self.__mean_background
            else:
                # If the background is absent, the dark current is generated with a given mean.
                dark_current_mean = dk_c

            self.dark_current = rng.normal(dark_current_mean, rel_var * dark_current_mean, self.shape)
            self.__has_dark_current = True

    def show_field(self, field='true'):
        """
        Method that plots an image of the field.

        Available `field` types are:
            - 'true':
                Plots the field of stars as they are outside the atmosphere. Note that this allows to print also the
                'mass' and 'magnitude' datatypes.
            - 'ph_noise':
                Plots the field for which the photon noise has been simulated.
            - 'background':
                Plots the field with the simulated background added.
            - 'psf':
                Plots the field convolved with the psf kernel of choice.
            - 'exposure':
                Plots the complete field after an observation, so with the combined effects of gain and dark current.
            - 'gain_map':
                Plots the gain map of the CCD.
            - 'dark_current':
                Plots the dark current over the CCD.

        Note that, to make the images coherent with the coordinates saved inside the `skysource.SkySource` objects,
        plots actually represent the transposed fields. You should apply `np.transpose` method to the field if you
        wished to obtain the same fields separately.

        Parameters
        ----------
        field: `str` in ['true', 'ph_noise', 'background', 'psf', 'exposure', 'gain_map', 'dark_current']
            The type of field to be plotted.

        Returns
        -------
        Shows a plot of the requested field.

        Raises
        ------
        `TypeError`:
            If `field` is not a string.
        `ValueError`:
            If `field` is not in the aforementioned list.
        `FieldAlreadyInitializedWarning`:
            If the dark current had been previously simulated. Execution is not halted, but the dark current is not
            updated.

        See Also
        --------
        `Field.initialize_field`, `Field.add_photon_noise`, `Field.add_background`, `Field.apply_psf`,
        `Field.create_gain_map`, `Field.create_dark_current`, `utils.DataType` and `numpy.transpose`.
        """
        if not isinstance(field, str):
            raise TypeError('`field` argument must be a string.')
        elif isinstance(field, str) and field not in self.__valid_fields:
            raise ValueError('`field` argument not in accepted list of valid fields.')

        image = getattr(self, self.__attributes[field])

        fig = plt.figure()
        ax1 = fig.gca()
        ax1.grid()

        im = ax1.imshow(image.T, origin='lower', cmap='binary')
        ax1.set_xlabel('RA')
        ax1.set_ylabel('DEC')

        cbar = fig.colorbar(im)
        cbar.ax.set_ylabel(self.datatype.capitalize())
        plt.show()

    def record_field(self, kernel, delta_time=1,
                     background_fluct='gauss', snr=10, bgnd_rel_var=0.05,
                     gain_mean=1, gain_rel_var=0.01,
                     dk_c_fraction=0.1, dk_c_rel_var=0.01, dk_c=1,
                     force=False):
        """
        Method that simulates the observation of a field.

        This method runs all the simulation steps one after another, with the initial parameters defined inside the
        arguments. If a gain map is still available another one CANNOT be generated from here, but must be manually
        generated with the appropriate method (See `Field.create_gain_map` method). Calling `G` the gain map,
        `F` the field convoluted with the psf kernel and `Id` the dark current of the CCD, the complete field is
        found as:

            Field.recorded_field = G * S + Id

        Parameters
        ----------
        kernel: `fieldsim.psf.Kernel`
            Kernel of the point spread function. See `fieldsim.psf.Kernel`.
        delta_time: `number`
            Factor that represents a longer or shorter exposure. Default is `1`.
        background_fluct: `str`
            Distribution that the background follows. At the moment is a placeholder, since it is always set as a
            gaussian. Default is `'gaussian'`.
        snr: `number`
            Estimate of the Signal to Noise Ratio, used to generate the background. Default is `10`.
        bgnd_rel_var: `number`
            Relative dispersion of the background distribution. Default is `0.05`.
        gain_mean: `number`
            Mean value of the gain of the CCD. Default is 1.
        gain_rel_var: `number`
            Relative dispersion of the gain of the CCD. Default is 0.01.
        dk_c_fraction: `number`
            Fraction of the mean background to be set as mean of the dark current of the CCD. Default is 0.1 (optional).
        dk_c_rel_var: `number >= 0
            Relative dispersion of the dark current of the CCD. Default is 0.01.
        dk_c: `number`
            Mean value of the dark current of the CCD, if no background is available. Default is 1 (optional).
        force: `bool`
            Flag to force re-initialization of the field if it had already been initialized. Default is `False`.

        Raises
        ------
        `TypeError`:
            If `force` is not `bool`.
        `NotInitializedError`:
            If the field has not been initialized, yet.
        `FieldAlreadyInitializedWarning`:
            If the exposure has been previously simulated. Execution is not halted, but the exposure is not simulated.

        See Also
        --------
        `Field.initialize_field`, `Field.add_photon_noise`, `Field.add_background`, `Field.apply_psf`,
        `Field.create_gain_map`, `Field.create_dark_current`.
        """
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
            # Simulation advancement at each step after initialization.
            self.add_photon_noise(delta_time=delta_time, force=True, multiply=False)
            self.add_background(fluct=background_fluct, snr=snr, rel_var=bgnd_rel_var, force=True)
            self.apply_psf(kernel=kernel, force=True)

            if not self.__has_gain_map:
                # if a gain map has not been generated, it is now, otherwise the generated one is used. Overriding
                # the generated gain map should be a careful decision because it is intrinsically tied to the CCD
                # production process and it is not a random variable, for a given CCD.
                self.create_gain_map(mean_gain=gain_mean, rel_var=gain_rel_var, force=True)
            self.create_dark_current(b_fraction=dk_c_fraction, rel_var=dk_c_rel_var, dk_c=dk_c, force=True)

            self.recorded_field = self.gain_map * self.w_psf_field + self.dark_current
            self.__has_record = True

    @property
    def shot(self):
        """
        Property of a `Field` instance, returns a "shot" of the actual `ImageStatus` of the field. See
        `utils.ImageStatus` for more information.
        """
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
