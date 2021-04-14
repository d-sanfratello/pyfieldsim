import numpy as np
import random as rd

from fieldsim.excep import WrongCoordsFormatError
from fieldsim.excep import WrongCoordsLengthError


class SkySource:
    def __init__(self, coords, m_min=1, m_max=350):
        try:
            iter(coords)
        except TypeError:
            raise WrongCoordsFormatError
        else:
            if len(coords) != 2:
                raise WrongCoordsLengthError

        self.coords = coords

        self.__m_min = m_min
        self.__m_max = m_max

        self.mass = None
        self.luminosity = None
        self.magnitude = None

    def initialize(self, e_imf=2.4, e_lm=3, cst_lm=1):
        self.mass = self.random_cimf(rd.random(), e=e_imf)
        self.luminosity = self.lm_relation(self.mass, e=e_lm, cst=cst_lm)
        self.magnitude = self.l2mag(self.luminosity)

        return self

    def random_cimf(self, cimf, e=2.4):
        if e <= 1:
            raise ValueError("IMF exponent cannot be smaller or equal to 1.")

        # setup a max value because, even if unlikely, irrealistically massive stars have been extracted.
        return ((self.m_max**(1 - e) - self.m_min**(1 - e)) * cimf + self.m_min**(1 - e))**(1 / (1 - e))

    @staticmethod
    def lm_relation(mass, e=3, cst=1):
        if e <= 0:
            raise ValueError("Luminosity increases with mass, usually.")

        return cst * mass ** (-e)

    @staticmethod
    def l2mag(luminosity):
        return -2.5 * np.log10(luminosity)

    @property
    def m_min(self):
        return self.__m_min

    @property
    def m_max(self):
        return self.__m_max
