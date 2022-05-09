from sympy import symbols

class HermiteInterpolation:

    def __init__(self):

        self.theta, self.y_n, self.y_np1, self.f_n, self.f_np1, self.h = \
            symbols("theta y_n y_np1 f_n f_np1 h")

        self.hermite_interpolation = (
            1 - self.theta) * self.y_n + self.theta * self.y_np1 + \
                self.theta * (self.theta - 1) * ((1 - 2 * self.theta) * (
                    self.y_np1 - self.y_n) + (self.theta - 1) * self.h * \
                        self.f_n + self.theta * self.h * self.f_np1)

    @staticmethod
    def calculate_hermite_interpolation(y_0, y_1, dydx_0, dydx_1, theta, h):
        return (1 - theta) * y_0 + theta * y_1 + theta * (theta - 1) * (
            (1- 2 * theta) * (y_1 - y_0) + (theta - 1) * h * dydx_0 + \
                theta * h * dydx_1)