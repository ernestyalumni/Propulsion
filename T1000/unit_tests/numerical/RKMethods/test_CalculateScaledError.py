from T1000.numerical.RKMethods.CalculateScaledError import CalculateScaledError
from T1000.numerical.RKMethods.DOPRI5Coefficients import DOPRI5Coefficients
from T1000.numerical.RKMethods.kCoefficients import kCoefficients
from unit_tests.numerical.RKMethods.fixtures import DOPRI5_fixture, \
    example_derivative, example_setup

import pytest

def test_calculate_scaled_error_constructs():
    epsilon = 10**(-6)
    scaled_err = CalculateScaledError(epsilon, 2 * epsilon)

    assert scaled_err.a_tolerance_ == 10**(-6)
    assert scaled_err.r_tolerance_ == 2 * 10**(-6)

def test_calculate_scaled_error_calculates(
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

def test_calculate_scaled_error_calculates_with_zero_atolerance(
        DOPRI5_fixture,
        example_setup):
    calc = DOPRI5_fixture
    setup = example_setup
    k_coefficients = kCoefficients(DOPRI5Coefficients.s, 1)

    epsilon = 10**(-6)

    scaled_err = CalculateScaledError(0.0, epsilon)

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

    assert result == 17.094490672479946

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

    assert result == 6.709464932952823

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

    assert result == 9.180963231338264


def test_calculate_scaled_error_calculates_with_larger_tolerances(
        DOPRI5_fixture,
        example_setup):
    calc = DOPRI5_fixture
    setup = example_setup
    k_coefficients = kCoefficients(DOPRI5Coefficients.s, 1)

    epsilon = 10**(-2)

    scaled_err = CalculateScaledError(0.0, epsilon)

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

    assert result == 0.0017094490672479945

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

    assert result == 0.0006709464932952822

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

    assert result == 0.0009180963231338264
