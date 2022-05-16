from T1000.numerical.RKMethods.CalculateScaledError import CalculateScaledError
from T1000.numerical.RKMethods.ComputePIStepSize import ComputePIStepSize
from T1000.numerical.RKMethods.DOPRI5Coefficients import DOPRI5Coefficients
from T1000.numerical.RKMethods.kCoefficients import kCoefficients
from unit_tests.numerical.RKMethods.fixtures import DOPRI5_fixture, \
    example_derivative, example_setup

import pytest

def test_computePIStepSize_constructs_with_default_values():

    return

def test_compute_PI_step_size_calculates(
        DOPRI5_fixture,
        example_setup):
    calc = DOPRI5_fixture
    setup = example_setup
    k_coefficients = kCoefficients(DOPRI5Coefficients.s, 1)

    epsilon = 10**(-6)

    scaled_err = CalculateScaledError(epsilon, epsilon)

    # Step 1

    y_out = calc.apply_method(
        setup.h,
        setup.x_0,
        setup.y_0,
        setup.dydx_0,
        k_coefficients)

    calculated_error = calc.calculate_error(setup.h, k_coefficients)

    result = scaled_err.calculate_scaled_error(
        1,
        setup.y_0,
        y_out,
        calculated_error)

    assert result == 10.047088008562323

    # Step 2

    y_in = y_out
    y_out = calc.apply_method(
        setup.h,
        setup.x_0 + setup.h,
        y_out,
        example_derivative(setup.x_0 + setup.h, y_out),
        k_coefficients)

    calculated_error = calc.calculate_error(setup.h, k_coefficients)

    result = scaled_err.calculate_scaled_error(
        1,
        y_in,
        y_out,
        calculated_error)

    assert result == 4.86664616926306

    # Step 3

    y_in = y_out
    y_out = calc.apply_method(
        setup.h,
        setup.x_0 + setup.h,
        y_out,
        example_derivative(setup.x_0 + setup.h, y_out),
        k_coefficients)

    calculated_error = calc.calculate_error(setup.h, k_coefficients)

    result = scaled_err.calculate_scaled_error(
        1,
        y_in,
        y_out,
        calculated_error)

    assert result == 7.554414512354117
