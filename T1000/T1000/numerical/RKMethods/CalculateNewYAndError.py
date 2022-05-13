class CalculateNewYAndError:

    def __init__(
            self,
            S,
            derivative,
            a_coefficients,
            c_coefficients,
            delta_coefficients):

        self._s = S
        self._derivative = derivative
        self._a_coefficients = a_coefficients
        self._c_coefficients = c_coefficients
        self._delta_coefficients = delta_coefficients

    def get_a_ij(self, i, j):

        return self._a_coefficients.get_ij_element(i, j)

    def get_c_i(self, i):

        return self._c_coefficients.get_ith_element(i)

    def _sum_a_and_k_products(self, k_coefficients, l, h):

        a_lj_times_k_j = k_coefficients.scalar_multiply(1, self.get_a_ij(l, 1))

        for j in range(2, l, 1):

            a_lj_times_k_j += k_coefficients.scalar_multiply(
                j,
                self.get_a_ij(l, j))

        return h * a_lj_times_k_j

    def apply_method(self, h, x, y, initial_dydx, k_coefficients):

        y_out = None

        k_coefficients._k_coefficients[0] = initial_dydx

        for l in range(2, self._s + 1):

            x_l =  x + self.get_c_i(l) * h

            y_out = self._sum_a_and_k_products(k_coefficients, l, h)

            y_out += y
            k_coefficients._k_coefficients[l - 1] = self._derivative(x_l, y_out)

        return y_out

    def calculate_error(self, h, k_coefficients):

        y_error = k_coefficients.scalar_multiply(
            1,
            self._delta_coefficients.get_ith_element(1))

        for j in range(2, self._s + 1):
            y_error += k_coefficients.scalar_multiply(
                j,
                self._delta_coefficients.get_ith_element(j))
        return h * y_error