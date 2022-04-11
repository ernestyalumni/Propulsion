import numpy as np

class aCoefficients:

    def __init__(self, s, a_coefficients):

        assert (s * (s - 1) / 2) == len(a_coefficients)
        # cf. https://www.statology.org/convert-list-to-numpy-array/
        self._a_coefficients = np.asarray(a_coefficients, dtype=np.float64)
        # cf. https://stackoverflow.com/questions/5541324/immutable-numpy-array
        # AttributeError: 'numpy.core.multiarray.flagsobj' object has no
        # attribute 'writable
        # self._a_coefficients.flags.writable = False
        self._s = s

    def get_ij_element(self, i, j):

        assert (i > j) and (j >= 1) and (i <= self._s)

        n = i - 2
        n = n * (n + 1) / 2
        return self._a_coefficients[int(n + (j - 1))]

    def get_ith_element(self, i):

        return self._a_coefficients[i]