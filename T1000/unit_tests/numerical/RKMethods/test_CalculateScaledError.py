from T1000.numerical.RKMethods.CalculateScaledError import CalculateScaledError
from T1000.numerical.RKMethods.DOPRI5Coefficients import DOPRI5Coefficients
from T1000.numerical.RKMethods.kCoefficients import kCoefficients
from unit_tests.numerical.RKMethods.fixtures import DOPRI5_fixture, example_setup

import pytest

def test_calculate_scaled_error_calculates(
        DOPRI5_fixture,
        example_setup):
    calc = DOPRI5_fixture
    setup = example_setup
    k_coefficients = kCoefficients(DOPRI5Coefficients.s, 1)

    y_out = calc.apply_method(
        setup.h,
        setup.x_0,
        setup.y_0,
        setup.dydx_0,
        k_coefficients)
