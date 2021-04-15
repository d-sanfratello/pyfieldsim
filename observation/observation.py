import numpy as np

from astropy.io import fits
from path import Path

from fieldsim.field import Field
from fieldsim.utils import ImageStatus
from fieldsim.utils import DataType
from fieldsim.utils import _status_values
from fieldsim.utils import _datatype_values

from fieldsim.excep import InvalidExtensionError
from fieldsim.excep import IncompatibleStatusError


class Observation:
    __lst_header = {'status': 1,
                    'datatype': 2}

    def __init__(self, field, ext='fits'):
        self.__field = field
        self.__ext = ext

        self.status = None
        self.datatype = None

        self.image = self.__extract_field(field, ext)

    def __extract_field(self, field, ext):
        if isinstance(field, Field):
            self.status = field.status
            self.datatype = field.datatype

            return field.shot
        elif isinstance(field, (str, Path)):
            if ext == 'fits':
                hdulist = fits.open(field)

                self.fits = [hdu for hdu in hdulist]
                image = self.fits[1].data
                hdulist.close()

                return image
            elif ext == 'txt':
                with open(field, 'r') as imfile:
                    lines = imfile.readlines()
                    header_list = lines[0].split()
                    self.status = _status_values[(header_list[self.__lst_header['status']])]
                    self.datatype = _datatype_values[(header_list[self.__lst_header['datatype']])]

                    image = np.array([[float(px) for px in line.split()] for line in lines[1:]])

                return image
            else:
                raise InvalidExtensionError
        else:
            raise IOError("Unknown field passed as argument. Must either be a `Field` object or a path.")

    def update_image(self):
        self.image = self.__extract_field(self.__field, self.__ext)

    def count_single_stars(self):
        if self.status != ImageStatus().SINGLESTARS:
            raise IncompatibleStatusError

        image_copy = self.image.copy()
        recorded_stars = []
        recorded_coords = []

        if self.datatype == DataType().MAGNITUDE:
            image_brighter = image_copy.min()
            while image_brighter < 0:
                coords = np.argmin(image_copy)
                coords = np.unravel_index(coords, image_copy.shape)

                recorded_stars.append(image_copy[coords])
                recorded_coords.append(coords)

                image_copy[coords] = 0
                image_brighter = image_copy.min()
        elif self.datatype == DataType().LUMINOSITY or self.datatype == DataType().MASS:
            image_brighter = image_copy.max()
            while image_brighter > 0:
                coords = np.argmax(image_copy)
                coords = np.unravel_index(coords, image_copy.shape)

                recorded_stars.append(image_copy[coords])
                recorded_coords.append(coords)

                image_copy[coords] = 0
                image_brighter = image_copy.max()

        recorded_stars = np.array(recorded_stars)
        recorded_coords = np.array(recorded_coords)

        return recorded_stars, recorded_coords
