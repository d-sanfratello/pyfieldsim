class ImageStatus:
    @property
    def NOTINIT(self):
        return -1

    @property
    def SINGLESTARS(self):
        return 'single_stars'


class DataType:
    @property
    def NOTINIT(self):
        return -1

    @property
    def LUMINOSITY(self):
        return 'luminosity'

    @property
    def MAGNITUDE(self):
        return 'magnitude'

    @property
    def MASS(self):
        return 'mass'


_NOINIT = -1
_status_values = {0: ImageStatus().SINGLESTARS}
_datatype_values = {0: DataType().LUMINOSITY,
                    1: DataType().MAGNITUDE,
                    2: DataType().MASS}
