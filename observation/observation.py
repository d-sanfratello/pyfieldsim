import numpy as np

from astropy.io import fits
from os.path import Path

from fieldsim.field import Field
from fieldsim.excep import InvalidExtensionError


class Observation:
    def __init__(self, field, ext='fits'):
        if isinstance(field, Field):
            self.image = field.shot
        elif isinstance(field, (str, Path)):
            if ext == 'fits':
                hdulist = fits.open(field)

                self.fits = [hdu for hdu in hdulist]
                self.image = self.fits[1].data

                hdulist.close()
            elif ext == 'txt':
                with open(field, 'r') as imfile:
                    self.image = np.array([[float(px) for px in line.split()] for line in imfile])
            else:
                raise InvalidExtensionError
        else:
            raise IOError("Unknown field passed as argument. Must either be a `Field` object or a path.")
