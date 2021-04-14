class ImageStatus:
    @property
    def NOTINIT(self):
        return -1

    @property
    def SINGLESTARS(self):
        return 0


class DataType:
    @property
    def NOTINIT(self):
        return -1

    @property
    def LUMINOSITY(self):
        return 0

    @property
    def MAGNITUDE(self):
        return 1

    @property
    def MASS(self):
        return 2
