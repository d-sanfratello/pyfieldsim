import numpy as np

from astropy.io import fits
from path import Path

from ..field import Field
from ..utils import ImageStatus
from ..utils import DataType
# noinspection PyProtectedMember
from ..utils import _status_values
# noinspection PyProtectedMember
from ..utils import _datatype_values

from ..excep import InvalidExtensionError
from ..excep import IncompatibleStatusError


class Observation:
    __lst_header = {'status': 1,
                    'datatype': 2}

    def __init__(self, field, ext='fits'):
        """
        Class that initializates an `Observation` instance, to simulate an exposure through a telescope and its
        following analysis.

        At initialization the class takes a `field` argument, either a `field.Field` object, a string to the path or
        a `Path` object and, in the last two cases, a extension `ext` being either `"fits"` or `"text"`. If `field`
        is a `Field` object, argument `ext` is ignored.

        Parameters
        ----------
        field: `field.Field`, string or `path.Path`
            Either a `Field` type object or the path to a .fits file or .txt file.
        ext: string either `"fits"` or `"txt"`
            If `field` is a path, the extension of the file to open. Default is `"fits"`.

        Raises
        ------
        `InvalidExtensionError`:
            If `ext` is not a known extension (`"txt"` or `"fits"`).
        `IOError`:
            If `field` is not a valid argument.

        See Also
        --------
        `field.Field.shot`.
        """
        self.__field = field
        self.__ext = ext

        self.status = None
        self.datatype = None

        self.image = self.__extract_field(field, ext)

    def __extract_field(self, field, ext):
        """
        Method that called at `observation.Observation` initialization and whenever the saved image needs to be
        updated. This method extracts an exposure from `field` for a set type of data file defined in `ext`.

        Returns
        -------
        shot: `numpy.ndarray`
            Returns the property `shot` of an instance of `field.Field`, if `field` is such instance, or the matrix
            representing the exposure as extracted from the .txt or the .fits file.
        """
        if isinstance(field, Field):
            self.status = field.status
            self.datatype = field.datatype

            return field.shot
        elif isinstance(field, (str, Path)):
            if ext == 'fits':
                # Still has to be corrected for status and datatype.
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
        """
        Method to update the image stored inside an `Observation` instance.

        In the case of an image generated from a `field.Field` object, any step in the simulation updates the
        `Field.shot` property. This method allows for the stored image to be update accordingly.

        Notice that this method works even if the image is extracted from a file, but it won't change the stored
        image unless the file was actually different, since it would simply re-read its content.

        See Also
        --------
        `field.Field.shot`.
        """
        self.image = self.__extract_field(self.__field, self.__ext)

    def count_single_stars(self):
        """
        Method that counts the stars in a given field, if the `ImageStatus` of the image is `SINGLESTARS` or `PH_NOISE`.

        Since any other step in the simulation adds background or spreads the values over different pixels,
        usage of this method is forbidden in those cases.

        This method stores a copy of the image extracted at initialization and finds the highest value (if
        `field.Field.datatype` is `LUMINOSITY` or `MASS`) or the lowest (if `MAGNITUDE`). It also stores its
        coordinates. So the pixel of the image copy at those coordinates is set to zero. This process is, then,
        reiterated until the field is filled with zeros and the values and the corrisponding coordinates are returned.

        Returns
        -------
        recorded_stars: `numpy.ndarray`
            Returns the array of measured values for the stars in the field. Those values are in the unit defined by
            the field datatype. See also `utils.DataType`.
        recorded_coords: `numpy.ndarray` of numpy arrays of length 2.
            Returns, for each value returned in `recorded_stars`, the coordinates at which that values has been found.

        Raises
        ------
        `IncompatibleStatusError`:
            If `field.Field.status` is not `SINGLESTARS` or `PH_NOISE`. As explained, this procedure wouldn't
            make sense with other steps of the simulation.

        See Also
        --------
        `field.Field`, `utils.DataType` and `utils.ImageStatus`.
        """
        if self.status not in [ImageStatus().SINGLESTARS, ImageStatus().PH_NOISE]:
            raise IncompatibleStatusError

        image_copy = self.image.copy()
        recorded_stars = []
        recorded_coords = []

        if self.datatype == DataType().MAGNITUDE:
            # In this implementation, the magnitude-like quantity cannot be greater than 0. So it is natural to look
            # for the minima.
            image_brighter = image_copy.min()
            while image_brighter < 0:
                coords = np.argmin(image_copy)
                coords = np.unravel_index(coords, image_copy.shape)

                recorded_stars.append(image_copy[coords])
                recorded_coords.append(coords)

                # Once the limit value has been found and stored, the corresponding pixel on the CCD is set to 0.
                image_copy[coords] = 0
                image_brighter = image_copy.min()
        elif self.datatype == DataType().LUMINOSITY or self.datatype == DataType().MASS:
            image_brighter = image_copy.max()
            while image_brighter > 0:
                coords = np.argmax(image_copy)
                coords = np.unravel_index(coords, image_copy.shape)

                recorded_stars.append(image_copy[coords])
                recorded_coords.append(coords)

                # once the limit value has been found and stored, the corresponding pixel on the CCD is set to 0.
                image_copy[coords] = 0
                image_brighter = image_copy.max()

        recorded_stars = np.array(recorded_stars)
        recorded_coords = np.array(recorded_coords)

        return recorded_stars, recorded_coords
