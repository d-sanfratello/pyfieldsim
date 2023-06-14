import os

import h5py
import numpy as np

from pathlib import Path

from pyfieldsim.utils.metadata import save_metadata

from pyfieldsim.errors.exceptions import WrongShapeError


class Sources:
    def __init__(self, shape):
        """
        Class that initializates a `Field` instance, to simulate an exposure
        through a telescope.

        At initialization, the class saves the shape of the output field. Every
        simulation step is described in its own method. See `dir(Field)` for the
        list of available methods.

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

        self.__pad = [self.shape[0] // 4, self.shape[1] // 4]
        self.ext_shape = tuple(
            [self.shape[_] + 2 * self.__pad[_] for _ in range(2)]
        )
        self.__n_pixels = np.asarray(self.ext_shape).prod()

        self.sources = None
        self.mass = None
        self.luminosity = None
        self.magnitude = None

        self.n_stars = None

        self.seed = None
        self.__rng = None

        self.__m_min = None
        self.__m_max = None

        self.__density = None
        self.__e_imf = None
        self.__e_lm = None
        self.__cst_lm = None

    def initialize_field(self,
                         m_min=1, m_max=350,
                         density=0.05, e_imf=2.4, e_lm=3, cst_lm=1,
                         seed=None):
        """
        Method that initializates the field randomly generating the stars in it
        as instances of the `skysource.SkySource` class.

        It defines an auxiliary field `Field.__aux_field` whose shape is the
        shape defined at initialization with a padding added. This will be
        useful at later stages of simulations, where the convolution with the
        psf kernel would generate boundary artifacts. In this implementation,
        though, the artifacts are far from the actual boundary of the image,
        which is realized by cropping the auxiliary field at its center. The
        size of the padding is, in each direction, equal to 1/4 of the
        corresponding edge's shape.

        After initializing the random number generator from numpy, the auxiliary
        field is initialized with random numbers in the interval [0, 1). If the
        generated number in a given pixel is below a threshold given by
        `density`, a star is initialized at those coordinates, by using the
        `skysource.SkySource` class.

        The auxiliary field is, then, cropped to obtain the actual field on the
        CCD. Also, in the `Field.sources` attribute the stars belonging to the
        padding area are deleted. This is to simulate their existence (and
        effect on the field when psf are taken into account) but without being
        able to measure their distance from the exposed field.

        There are three possible settings for the field initialization as
        `datatype`: 'luminosity', 'mass' and 'magnitude', where the last two are
        mainly for debugging reasongs. Every step of the simulation, indeed,
        require the `datatype` to be 'luminosity', as defined in
        `utils.DataType` class.

        Parameters
        ----------
        density: `number` in [0, 1]
            Expected number density of stars in the field. Default is `0.05`.
        e_imf: `number`
            Exponent of the IMF powerlaw, to be passed to the
            `skysource.SkySource.random_cimf` method. Default is `2.4`.
        e_lm: `number`
            Exponent of the mass-luminosity relation powerlaw, to be passed to
            the `skysource.SkySource.lm_relation` method. Default is `3`.
        cst_lm: `number`
            Factor in front of the mass-luminosity relation powerlaw, to be
            passed to the `skysource.SkySource.lm_relation` method. Default is
            `1`.
        seed: `int` or `None`
            If of type `int`, passes the seed to the pseudo-random number
            generator of numpy. If `None`, numpy generates a seed by itself.
            Default is `None`.

        Raises
        ------
        `TypeError`:
            If `datatype` is not a string or if `force` is not a `bool` or if
            `seed` is not an `int` or `None`.
        `ValueError`:
            If `datatype` is not `'luminosity'`, `'mass'` or `'magnitude'`.
        `FieldAlreadyInitializedWarning`:
            If the field has been previously initialized. Execution is not
            halted, but the field is not updated.
        """
        if e_imf <= 1:
            raise ValueError("IMF exponent cannot be smaller or equal to 1.")
        if cst_lm <= 0:
            raise ValueError("`cst` must be greater than zero.")
        if m_min >= m_max:
            raise ValueError("`m_max` has to be strictly greater than "
                             "`m_min`.")

        self.seed = seed
        self.__rng = np.random.default_rng(seed=self.seed)

        self.__m_min = m_min
        self.__m_max = m_max

        self.__density = density
        self.__e_imf = e_imf
        self.__e_lm = e_lm
        self.__cst_lm = cst_lm

        # Random generation of the stars
        rand_dist_loc = self.__rng.random(self.__n_pixels)
        coords_arr = np.argwhere(rand_dist_loc <= density).flatten()
        rand_dist_mas = self.__rng.random(coords_arr.shape)

        self.sources = np.asarray(
            np.unravel_index(coords_arr, self.ext_shape)
        ).T

        # Generating masses
        self.mass = self.__random_cimf(rand_dist_mas)
        self.luminosity = self.__lm_relation()
        self.magnitude = self.__l2mag()

        self.n_stars = self.sources.shape[0]

    def __random_cimf(self, cimf):
        """
        Method that generates a mass from a powerlaw IMF, given a randomly
        generated value in the interval [0, 1].

        It uses the cdf of a powerlaw defined on the interval [`m_min`,
        `m_max`], where the limits are given at instance creation. The mass
        corresponding to a given `cimf` is given by:

                 /                                                     \ 1/(1-e)
            M = |[M_max^(1 - e) - M_min^(1 - e)] * cimf + M_min^(1 - e)|
                \                                                     /

        Parameters
        ----------
        cimf: `number` in [0, 1]
            Value of the cdf to generate a mass from.
        e: `number` > 1
            Exponent of the IMF powerlaw. Default is `2.4`.

        Returns
        -------
        Mass: `number`
            Returns the mass of the star.

        Raises
        ------
        `ValueError`:
            If `cimf` is not in the iterval [0, 1] or if `e` is less or
            equal to 1.
        """
        if not np.all((0 <= cimf) & (cimf <= 1)):
            raise ValueError("`cimf` value is outside of interval [0, 1].")

        e = self.__e_imf

        # A max value was set because, even if unlikely, irrealistically
        # massive stars have been extracted.
        const = self.__m_max**(1 - e) - self.__m_min**(1 - e)
        min_exp = self.__m_min**(1 - e)

        return (const * cimf + min_exp)**(1 / (1 - e))

    def __lm_relation(self):
        """
        Method to determine the luminosity of a star given its mass and a
        powerlaw L-M relation.

        It calculates the luminosity of the star from a relation given by:

            L = cst * M^e

        Parameters
        ----------
        e: `number` > 0
            Exponent of the L-M powerlaw. Default is `3`.
        cst: `number` > 0
            Constant multiplying the L-M relation. Default is `1`.

        Returns
        -------
        Luminosity: `number`
            Returns the luminosity of the star.

        Raises
        ------
        `ValueError`:
            If `e` is less or equal to 0 or if `cst` is less or equal to 0.
        `AttributeError`:
            If instance's mass has not been generated, yet.
        """
        return self.__cst_lm * self.mass ** self.__e_lm

    def __l2mag(self):
        """
        Method to determine a magnitude-like quantity of a star given its
        luminosity.

        It calculates a magnitude-like quantity with the following relation:

            mag = -2.5 * log10(L)

        Returns
        -------
        Magnitude-like: `number`
            Returns a magnitude-like quantity for the star.

        Raises
        ------
        `AttributeError`:
            If instance's luminosity has not been determined, yet.
        """
        return -2.5 * np.log10(self.luminosity)

    def export_sources(self, filename):
        filename = Path(filename)
        if filename.suffix.lower() != '.h5':
            filename = filename.with_suffix('.h5')

        with h5py.File(filename, "w") as file:
            coords_dset = file.create_dataset(
                'coords',
                shape=self.sources.shape,
                dtype=np.int
            )
            coords_dset[0:] = self.sources

            mass_dset = file.create_dataset(
                'mass',
                shape=self.n_stars,
                dtype=np.double
            )
            mass_dset[0:] = self.mass

            lum_dset = file.create_dataset(
                'luminosity',
                shape=self.n_stars,
                dtype=np.double
            )
            lum_dset[0:] = self.mass

            mag_dset = file.create_dataset(
                'magnitude',
                shape=self.n_stars,
                dtype=np.double
            )
            mag_dset[0:] = self.magnitude

            save_metadata(
                metadata=self.metadata,
                filename=filename
            )

    @property
    def metadata(self):
        meta = {
            'shape': self.shape,
            'pad': self.__pad,
            'ext_shape': self.ext_shape,
            'seed': self.seed,
            'density': self.__density,
            'e_imf': self.__e_imf,
            'e_lm': self.__e_lm,
            'cst_lm': self.__cst_lm
        }

        return meta
