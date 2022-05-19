from T1000.numerical.RKMethods.CalculateNewYAndError \
    import CalculateNewYAndError
from T1000.numerical.RKMethods.DOPRI5Coefficients import DOPRI5Coefficients
from collections import namedtuple

import numpy as np
import pytest

@pytest.fixture
def DOPRI5_fixture():
    calc = CalculateNewYAndError(
        DOPRI5Coefficients.s,
        example_derivative,
        DOPRI5Coefficients.a_coefficients,
        DOPRI5Coefficients.c_coefficients,
        DOPRI5Coefficients.delta_coefficients)

    return calc    

@pytest.fixture
def example_setup():

    return namedtuple(
        "ExampleSetup",
        ["x_0", "y_0", "dydx_0", "h", "previous_error"])(
            0.0,
            np.array([0.5,]),
            np.array([1.5,]),
            0.5,
            1e-4)


def example_derivative(t, y):
    """
    @param k [in/out] k serves as the output.
    """
    return y - t * t + 1.0


def example_exact_solution(t):
    return t * t + 2 * t + 1 - 0.5 * np.exp(t)
