class aCoefficients:

    def __init__(self, s, a_coefficients):

        assert (s * (s - 1) / 2) == len(a_coefficients)
        self._a_coefficients = a_coefficients
        self._s = s

    def get_ij_element(self, i, j):

        assert (i > j) and (j >= 1) and (i <= self._s)

        n = i - 2
        n = n * (n + 1) / 2
        return self._a_coefficients[n + (j - 1)]

    def get_ith_element(self, i):

        return self._a_coefficients[i]