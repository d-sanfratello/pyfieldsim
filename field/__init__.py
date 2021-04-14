from .field import Field


class ImageStatus:
    @property
    def NOTINIT(self):
        return -1

    @property
    def SINGLESTARS(self):
        return 0
