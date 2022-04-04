class bCoefficients:

    def __init__(self, s, b_coefficients):

        assert s == len(b_coefficients)
        self._b_coefficients = b_coefficients
        self._s = s

    def get_ith_element(self, i):

        return self._b_coefficients[i - 1]