import numpy as np

class RungeKuttaMethod:

    def __init__(
            self,
            m,
            alpha_coefficients,
            beta_coefficients,
            c_coefficients):
        self._m = m
        self._alpha_coefficients = alpha_coefficients
        self._beta_coefficients = beta_coefficients
        self._c_coefficients = c_coefficients

    def get_beta_ij(self, i, j):
        n = (i - 2)
        n = n * (n + 1) / 2

        return self._beta_coefficients[int(n) + (j - 1)]

    def get_alpha_i(self, i):
        return self._alpha_coefficients[i - 2]

    def get_c_i(self, i):
        return self._c_coefficients[i - 1]

    def _sum_beta_and_k_products(self, k_coefficients, l, h):

        summation = self.get_beta_ij(l, 1) * k_coefficients[0]

        for j in range(2, l):
            summation += self.get_beta_ij(l, j) * k_coefficients[j - 1]

        return h * summation

    def _calculate_k_coefficients(self, x_n, t_n, h, f):
        k_coefficients = []
        k_coefficients.append(f(t_n, x_n))

        for l in range(2, self._m + 1):

            t_l = t_n + h * self.get_alpha_i(l)

            x_l = self._sum_beta_and_k_products(k_coefficients, l, h)
            x_l += x_n

            k_coefficients.append(f(t_l, x_l))

        return k_coefficients

    def calculate_next_step(self, x_n, t_n, h, f):

        k_coefficients = self._calculate_k_coefficients(x_n, t_n, h, f)

        x_np1 = np.sum(
            [c_i * k_i
                for c_i, k_i in zip(self._c_coefficients, k_coefficients)],
            axis=0)

        return x_n + h * x_np1

