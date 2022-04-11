import numpy as np

class kCoefficients:

    def __init__(self, s, n):

        self._k_coefficients = [np.zeros(n) for i in range(s)]
        self._s = s
        self._n = n

    def get_ith_coefficient(self, i):

        return self._k_coefficients[i - 1]

    def scalar_multiply(self, k_coefficient_index, scalar):

        return self._k_coefficients[k_coefficient_index - 1] * scalar

    def size(self):
        return len(self._k_coefficients)

    def dimensions(self):
        return self._k_coefficients[0].shape