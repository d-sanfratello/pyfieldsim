class WrongShapeError(Exception):
    def __str__(self):
        return "Shape of sky field is incorrect. Must be a `tuple` of length 2."


class WrongCoordsFormatError(Exception):
    def __str__(self):
        return "Coordinates for source are not iterable, check them."


class WrongCoordsLengthError(Exception):
    def __str__(self):
        return "Coordinates for source are not 2D coordinates."


class InvalidExtensionError(Exception):
    def __str__(self):
        return "Invalid file extension. Must be either 'txt' or 'fits'."


class IncompatibleStatusError(Exception):
    def __str__(self):
        return "Operation requeste is incompatible with the status of the image."


class NotInitializedError(Exception):
    def __str__(self):
        return "Field has not been initialized."
