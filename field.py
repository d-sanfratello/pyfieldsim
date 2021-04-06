import numpy as np

from fieldsim.exceptions import WrongShapeError


class Field:
    def __init__(self, shape):
        if not isinstance(shape, tuple) or len(a) != 2:
            raise WrongShapeError("Shape of sky field is incorrect. Must be a `tuple` of length 2.")

        self.shape = shape
