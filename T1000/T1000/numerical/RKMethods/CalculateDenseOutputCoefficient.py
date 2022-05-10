class CalculateDenseOutputCoefficient:

    def __init__(self, S, dense_coefficients):
        self._s = S
        self._dense_coefficients = dense_coefficients

    def calculate_dense_output_coefficient(self, k_coefficients, theta, h):

        dense_output_coefficient = \
            self._dense_coefficients.get_ith_element(1) * \
                k_coefficients._k_coefficients[0]

        for l in range(2, self._s + 1):
            
            dense_output_coefficient += \
                self._dense_coefficients.get_ith_element(l) * \
                    k_coefficients._k_coefficients[l - 1]

        return theta * theta * (1.0 - theta) * (1.0 - theta) * h * \
          dense_output_coefficient
