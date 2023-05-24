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
