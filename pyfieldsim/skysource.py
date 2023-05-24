import numpy as np

from pyfieldsim.utils.exceptions import WrongCoordsFormatError
from pyfieldsim.utils.exceptions import WrongCoordsLengthError


class SkySource:
    def __init__(self, coords, m_min=1, m_max=350):
        """
        Class that initializates a `SkySource` instance, generating its mass
        with a given IMF and determining its luminosity, given a mass-luminosity
        relation.

        At initialization, the class saves the coordinates of the source
        (relative to the field) and the mass limits to be used for a powerlaw
        Initial Mass Function.

        Parameters
        ----------
        coords: `iterable` of size 2
            Coordinates of the source within the field.
        m_min: `number`
            Lowest limit of mass for the IMF. Default is 1.
        m_max: `number`
            Highest limit of mass for the IMF. Default is 350.

        Raises
        ------
        `WrongCoordsFormatError`:
            If `coords` is not an iterable object.
        `WrongCoordsLengthError`:
            If `coords` is not an iterable of length 2.

        Examples
        --------
        >>> # Generates a source in field location (2,5), whose mass is to be
        >>> # defined whitin 1 and 100 arbitrary mass units.
        >>> src = SkySource((2,5), m_min=1, m_max=100)

        >>> # Generates a source in the same location, but with default mass
        >>> # boundaries and location defined with a numpy array.
        >>> src = SkySource(np.nrdarray([2,5]))
        """
        try:
            iter(coords)
        except TypeError:
            raise WrongCoordsFormatError
        else:
            if len(coords) != 2:
                raise WrongCoordsLengthError

        self.coords = np.array(coords)

        self.__m_min = m_min
        self.__m_max = m_max

        self.mass = None
        self.luminosity = None
        self.magnitude = None

    def initialize(self, e_imf=2.4, e_lm=3, cst_lm=1, rng=None):
        """
        Method that initializates the mass, luminosity and magnitude-like
        quantities for a `SkySource`.

        It uses a random number generator from numpy to generate a random number
        in the [0, 1) interval. This value is, then, passed to the
        `self.random_cimf` method, to extract a mass in the [`self.m_min`,
        `self.m_max`) interval. Luminosity is, then, derived with a powerlaw
        defined in the `self.lm_relation` method. A magnitude-like quantity,
        defined in the `self.l2mag` method.

        Parameters
        ----------
        e_imf: `number`
            Exponent of the IMF powerlaw, to be passed to the `self.random_cimf`
            method. Default is `2.4`.
        e_lm: `number`
            Exponent of the mass-luminosity relation powerlaw, to be passed to
            the `self.lm_relation` method. Default is `3`.
        cst_lm: `number`
            Factor in front of the mass-luminosity relation powerlaw, to be
            passed to the `self.lm_relation` method. Default is `1`.
        rng: `numpy.random.Generator` or `None`
            If of type `numpy.random.Generator`, it uses the given pseudo-random
            number generator. If `None`, the `numpy.random.default_rng()`
            generator is used with a casual seed. Default is `None`. (Modified
            on May 5th 2021)

        Returns
        -------
        self: `SkySource`
            Returns the object.

        Raises
        ------
        `TypeError`:
            If `rng` is not an `np.random.Generator` or `None`. (Modified on May
            5th 2021)

        Examples
        --------
        >>> # Initializes the mass of the source with the Salpeter IMF. The
        >>> # luminosity powerlaw is set for a Main Sequence star, but the
        >>> # factor of the L-M relation is 100, to obtain higher values for the
        >>> # luminosity.
        >>> src.initialize(e_imf=2.35, e_lm=3, cst_lm=100)

        See Also
        --------
        `SkySource.random_cimf`, `SkySource.lm_relation`, `SkySource.l2mag` and
        `numpy.random.default_rng` or `numpy.random.Generator`. (Modified on May
        5th 2021)
        """
        if not isinstance(rng, np.random.Generator) and rng is not None:
            raise TypeError("`rng` argument must be either a "
                            "`numpy.random.Generator` or `None`.")

        if rng is None:
            rng = np.random.default_rng()
        else:
            rng = rng

        self.mass = self.random_cimf(rng.random(), e=e_imf)
        self.luminosity = self.lm_relation(e=e_lm, cst=cst_lm)
        self.magnitude = self.l2mag()

        return self

    def random_cimf(self, cimf, e=2.4):
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
            If `cimf` is not in the iterval [0, 1] or if `e` is less or equal to
            1.

        Examples
        --------
        >>> # Mass of a star following an IMF defined by the powerlaw M^-2.4, in
        >>> # case the random variable is 0.3, for for the default mass limits.
        >>> src.random_cimf(0.3, e=2.4)
        1.2900536896433321

        >>> # Mass of a star following an IMF defined by the powerlaw M^-2.4,
        >>> # in case the random variable is 0.6, for for the default mass
        >>> # limits.
        >>> src.random_cimf(0.6, e=2.4)
        1.9236020460960432

        >>> # Mass of a star following an IMF defined by the powerlaw M^-2.4, in
        >>> # case the random variable is 0.95, for for the default mass limits.
        >>> src.random_cimf(0.95, e=2.4)
        8.46631304311171

        See Also
        --------
        `SkySource`.
        """
        if not 0 <= cimf <= 1:
            raise ValueError("`cimf` value is outside of interval [0, 1].")
        if e <= 1:
            raise ValueError("IMF exponent cannot be smaller or equal to 1.")

        # A max value was set because, even if unlikely, irrealistically massive
        # stars have been extracted.
        return ((self.m_max**(1 - e) - self.m_min**(1 - e)) *
                cimf + self.m_min**(1 - e))**(1 / (1 - e))

    def lm_relation(self, e=3, cst=1):
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

        Examples
        --------
        >>> # Luminosity of a star of mass 2.5 arb. mass. units, for a relation
        >>> # L = M^3.
        >>> src.lm_relation(e=3, cst=1)
        15.625
        """
        if e <= 0:
            raise ValueError("Luminosity increases with mass, usually.")
        if cst <= 0:
            raise ValueError("`cst` must be greater than zero.")

        if self.mass is None:
            raise AttributeError("Star must be generated, before luminosity can"
                                 " be defined.")

        return cst * self.mass ** e

    def l2mag(self):
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

        Examples
        --------
        >>> # If the star has a luminosity of 10 arb. lum. units, this method
        >>> # returns:
        >>> src.l2mag()
        -2.5
        """
        if self.luminosity is None:
            raise AttributeError("Star must be generated and have a luminosity"
                                 " defined, before magnitude can be defined.")

        return -2.5 * np.log10(self.luminosity)

    @property
    def m_min(self):
        """
        Property of a `SkySource` type object. It is the lower mass limit for
        the IMF.
        """
        return self.__m_min

    @property
    def m_max(self):
        """
        Property of a `SkySource` type object. It is the higher mass limit for
        the IMF.
        """
        return self.__m_max
