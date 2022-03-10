from .data_type import DataType
from .image_status import ImageStatus
from .exceptions import *
from .warnings import *

_NOINIT = -1
_status_values = {0: ImageStatus().SINGLESTARS,
                  1: ImageStatus().PH_NOISE,
                  2: ImageStatus().BACKGROUND}
_datatype_values = {0: DataType().LUMINOSITY,
                    1: DataType().MAGNITUDE,
                    2: DataType().MASS}
