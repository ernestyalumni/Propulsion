import numpy as np

class cCoefficients:

    def __init__(self, s, c_coefficients):

        assert (s - 1) == len(c_coefficients)

        # cf. https://www.statology.org/convert-list-to-numpy-array/
        self._c_coefficients = np.asarray(c_coefficients, dtype=np.float64)

        # cf. https://stackoverflow.com/questions/5541324/immutable-numpy-array
        # AttributeError: 'numpy.core.multiarray.flagsobj' object has no
        # attribute 'writable
        #self._c_coefficients.flags.writable = False
        self._s = s

    def get_ith_element(self, i):

        return self._c_coefficients[i - 2]