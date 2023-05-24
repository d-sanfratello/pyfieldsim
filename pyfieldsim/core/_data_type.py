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
