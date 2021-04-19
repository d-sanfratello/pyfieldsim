class ImageStatus:
    """
    Class defining the status of the simulation of a field.
    """
    @property
    def NOTINIT(self):
        """
        Simulation has not been initialized, yet.
        """
        return -1

    @property
    def SINGLESTARS(self):
        """
        Simulation has been initialized with the stars' generation.
        """
        return 'single_stars'

    @property
    def PH_NOISE(self):
        """
        A field containing photon noise has been generated.
        """
        return 'ph_noise'

    @property
    def BACKGROUND(self):
        """
        A field containing background has been generated.
        """
        return 'background'

    @property
    def PSF(self):
        """
        A psf kernel has been applied to the field.
        """
        return 'psf'

    @property
    def EXPOSURE(self):
        """
        The simulation contains the whole exposure, taking into account the gain map and the dark current of the CCD.
        """
        return 'exposure'


class DataType:
    """
    Class defining the data type in the simulation.
    """
    @property
    def NOTINIT(self):
        """
        Simulation has not been initialized, yet.
        """
        return -1

    @property
    def LUMINOSITY(self):
        """
        Data represents the recorded luminosity in arb. lum. units. This is the datatype used throughout the simulation.
        """
        return 'luminosity'

    @property
    def MAGNITUDE(self):
        """
        Data represents a magnitude-like quantity. Mainly for debugging purposes, so far.
        """
        return 'magnitude'

    @property
    def MASS(self):
        """
        Data represents the mass of the generated stars. Mainly for debugging purposes.
        """
        return 'mass'


_NOINIT = -1
_status_values = {0: ImageStatus().SINGLESTARS,
                  1: ImageStatus().PH_NOISE,
                  2: ImageStatus().BACKGROUND}
_datatype_values = {0: DataType().LUMINOSITY,
                    1: DataType().MAGNITUDE,
                    2: DataType().MASS}
