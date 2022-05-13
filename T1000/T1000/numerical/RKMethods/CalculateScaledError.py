from math import pow, sqrt

class CalculateScaledError:

    def __init__(self, atol, rtol):
        self.a_tolerance_ = atol
        self.r_tolerance_ = rtol

    def calculate_scaled_error(self, N, y_0, y_out, y_err):

        error = 0.0

        for i in range(N):
            scale = self.a_tolerance_ + \
                self.r_tolerance_ * max(abs(y_0[i]), abs(y_out[i]))
            error += pow( y_err[i] / scale, 2)

        return sqrt(error / N)