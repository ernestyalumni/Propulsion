from T1000.numerical.RKMethods.CalculateScaledError import CalculateScaledError
from T1000.numerical.RKMethods.ComputePIStepSize import ComputePIStepSize
from T1000.numerical.RKMethods.DOPRI5Coefficients import DOPRI5Coefficients
from T1000.numerical.RKMethods.kCoefficients import kCoefficients
from unit_tests.numerical.RKMethods.fixtures import DOPRI5_fixture, \
    example_derivative, example_exact_solution, example_setup

import pytest

def test_calculate_recommended_alpha_beta_calculates():
    a, b = ComputePIStepSize.calculate_recommended_alpha_beta(5)
    assert a == 0.13999999999999999
    assert b == 0.08

    a, b = ComputePIStepSize.calculate_recommended_alpha_beta(8)
    assert a == 0.0875
    assert b == 0.05


def test_computePIStepSize_constructs_with_default_values():

    pi_step = ComputePIStepSize(0.7 / 5, 0.08)
    assert pi_step.alpha == 0.7 / 5
    assert pi_step.beta == 0.08
    assert pi_step.safety_factor == 0.9
    assert pi_step.min_scale == 0.2
    assert pi_step.max_scale == 5.0


def test_compute_PI_step_size_calculates(
        DOPRI5_fixture,
        example_setup):
    calc = DOPRI5_fixture
    setup = example_setup
    k_coefficients = kCoefficients(DOPRI5Coefficients.s, 1)

    pi_step = ComputePIStepSize(0.7 / 5, 0.08)

    epsilon = 10**(-6)

    scaled_err = CalculateScaledError(epsilon, epsilon)

    previous_error = setup.previous_error

    # Step 1

    y_out = calc.apply_method(
        setup.h,
        setup.x_0,
        setup.y_0,
        setup.dydx_0,
        k_coefficients)

    calculated_error = calc.calculate_error(setup.h, k_coefficients)

    error = scaled_err.calculate_scaled_error(
        1,
        setup.y_0,
        y_out,
        calculated_error)

    assert error == 10.047088008562323

    new_h = pi_step.compute_new_step_size(error, previous_error, setup.h, False)

    assert new_h == 0.3257818497635847

    y_out = calc.apply_method(
        new_h,
        setup.x_0,
        setup.y_0,
        setup.dydx_0,
        k_coefficients)

    calculated_error = calc.calculate_error(new_h, k_coefficients)

    error = scaled_err.calculate_scaled_error(
        1,
        setup.y_0,
        y_out,
        calculated_error)

    assert error == 1.517043286161648

    new_h = pi_step.compute_new_step_size(error, previous_error, new_h, True)

    assert new_h == 0.2765856717210728

    y_out = calc.apply_method(
        new_h,
        setup.x_0,
        setup.y_0,
        setup.dydx_0,
        k_coefficients)

    assert y_out[0] == pytest.approx(
        example_exact_solution(setup.x_0 + new_h),
        0.000001)

    calculated_error = calc.calculate_error(new_h, k_coefficients)

    error = scaled_err.calculate_scaled_error(
        1,
        setup.y_0,
        y_out,
        calculated_error)

    assert error == 0.7192693159670872

    x_in = setup.x_0 + new_h

    new_h = pi_step.compute_new_step_size(error, previous_error, new_h, True)

    assert new_h == 0.12476919493370708

    previous_error = error

    # Step 2

    y_in = y_out
    y_out = calc.apply_method(
        new_h,
        x_in,
        y_out,
        example_derivative(x_in, y_out),
        k_coefficients)

    assert y_out[0] == pytest.approx(
        example_exact_solution(x_in + new_h),
        0.000001)

    calculated_error = calc.calculate_error(new_h, k_coefficients)

    error = scaled_err.calculate_scaled_error(
        1,
        y_in,
        y_out,
        calculated_error)

    assert error == 0.01123118086582314

    x_in += new_h

    new_h = pi_step.compute_new_step_size(error, previous_error, new_h, False)

    assert new_h == 0.20504144131432045

    previous_error = error

    # Step 3

    y_in = y_out
    y_out = calc.apply_method(
        new_h,
        x_in,
        y_out,
        example_derivative(x_in, y_out),
        k_coefficients)

    assert y_out[0] == pytest.approx(
        example_exact_solution(x_in + new_h),
        0.000001)

    calculated_error = calc.calculate_error(new_h, k_coefficients)

    error = scaled_err.calculate_scaled_error(
        1,
        y_in,
        y_out,
        calculated_error)

    assert error == 0.09858701685826891

    new_h = pi_step.compute_new_step_size(error, previous_error, new_h, False)

    assert new_h == 0.1782310754175909


def test_compute_PI_step_size_calculates_with_scaled_error_of_zero_atolerance(
        DOPRI5_fixture,
        example_setup):
    calc = DOPRI5_fixture
    setup = example_setup
    k_coefficients = kCoefficients(DOPRI5Coefficients.s, 1)

    pi_step = ComputePIStepSize(0.7 / 5, 0.08)

    epsilon = 10**(-6)

    scaled_err = CalculateScaledError(0.0, epsilon)

    previous_error = setup.previous_error

    # Step 1

    y_out = calc.apply_method(
        setup.h,
        setup.x_0,
        setup.y_0,
        setup.dydx_0,
        k_coefficients)

    calculated_error = calc.calculate_error(setup.h, k_coefficients)

    error = scaled_err.calculate_scaled_error(
        1,
        setup.y_0,
        y_out,
        calculated_error)

    assert error == 17.094490672479946

    new_h = pi_step.compute_new_step_size(error, previous_error, setup.h, False)

    assert new_h == 0.3024214912553718

    y_out = calc.apply_method(
        new_h,
        setup.x_0,
        setup.y_0,
        setup.dydx_0,
        k_coefficients)

    calculated_error = calc.calculate_error(new_h, k_coefficients)

    error = scaled_err.calculate_scaled_error(
        1,
        setup.y_0,
        y_out,
        calculated_error)

    assert error == 2.1434187904795894

    new_h = pi_step.compute_new_step_size(error, previous_error, new_h, True)

    assert new_h == 0.24462462635573054

    y_out = calc.apply_method(
        new_h,
        setup.x_0,
        setup.y_0,
        setup.dydx_0,
        k_coefficients)

    assert y_out[0] == pytest.approx(
        example_exact_solution(setup.x_0 + new_h),
        0.0000001)

    calculated_error = calc.calculate_error(new_h, k_coefficients)

    error = scaled_err.calculate_scaled_error(
        1,
        setup.y_0,
        y_out,
        calculated_error)

    assert error == 0.8562538508592878

    x_in = setup.x_0 + new_h

    new_h = pi_step.compute_new_step_size(error, previous_error, new_h, True)

    assert new_h == 0.10769073212546879

    previous_error = error

    # Step 2

    y_in = y_out
    y_out = calc.apply_method(
        new_h,
        x_in,
        y_out,
        example_derivative(x_in, y_out),
        k_coefficients)

    assert y_out[0] == pytest.approx(
        example_exact_solution(x_in + new_h),
        0.0000001)

    calculated_error = calc.calculate_error(new_h, k_coefficients)

    error = scaled_err.calculate_scaled_error(
        1,
        y_in,
        y_out,
        calculated_error)

    assert error == 0.010974841057994985

    x_in += new_h

    new_h = pi_step.compute_new_step_size(error, previous_error, new_h, False)

    assert new_h == 0.1800417760399134

    previous_error = error

    # Step 3

    y_in = y_out
    y_out = calc.apply_method(
        new_h,
        x_in,
        y_out,
        example_derivative(x_in, y_out),
        k_coefficients)

    assert y_out[0] == pytest.approx(
        example_exact_solution(x_in + new_h),
        0.0000001)

    calculated_error = calc.calculate_error(new_h, k_coefficients)

    error = scaled_err.calculate_scaled_error(
        1,
        y_in,
        y_out,
        calculated_error)

    assert error == 0.09620632689256294

    new_h = pi_step.compute_new_step_size(error, previous_error, new_h, False)

    assert new_h == 0.15674696777373567

    previous_error = error


def test_compute_PI_step_size_calculates_scaled_errors_of_larger_tolerances(
        DOPRI5_fixture,
        example_setup):
    calc = DOPRI5_fixture
    setup = example_setup
    k_coefficients = kCoefficients(DOPRI5Coefficients.s, 1)

    pi_step = ComputePIStepSize(0.7 / 5, 0.08)

    epsilon = 10**(-2)

    scaled_err = CalculateScaledError(0.0, epsilon)

    previous_error = setup.previous_error

    # Step 1

    y_out = calc.apply_method(
        setup.h,
        setup.x_0,
        setup.y_0,
        setup.dydx_0,
        k_coefficients)

    assert y_out[0] == pytest.approx(
        example_exact_solution(setup.x_0 + setup.h),
        0.00001)

    calculated_error = calc.calculate_error(setup.h, k_coefficients)

    error = scaled_err.calculate_scaled_error(
        1,
        setup.y_0,
        y_out,
        calculated_error)

    assert error == 0.0017094490672479945

    x_in = setup.x_0 + setup.h

    new_h = pi_step.compute_new_step_size(error, previous_error, setup.h, False)

    assert new_h == 0.525548318135207

    previous_error = error

    # Step 2

    y_in = y_out
    y_out = calc.apply_method(
        new_h,
        x_in,
        y_out,
        example_derivative(x_in, y_out),
        k_coefficients)

    assert y_out[0] == pytest.approx(
        example_exact_solution(x_in + new_h),
        0.00001)

    calculated_error = calc.calculate_error(new_h, k_coefficients)

    error = scaled_err.calculate_scaled_error(
        1,
        y_in,
        y_out,
        calculated_error)

    assert error == 0.0008273888382715166

    x_in += new_h

    new_h = pi_step.compute_new_step_size(error, previous_error, new_h, False)

    assert new_h == 0.7673690892236448

    previous_error = error

    # Step 3

    y_in = y_out
    y_out = calc.apply_method(
        new_h,
        x_in,
        y_out,
        example_derivative(x_in, y_out),
        k_coefficients)

    assert y_out[0] == pytest.approx(
        example_exact_solution(x_in + new_h),
        0.0001)

    calculated_error = calc.calculate_error(new_h, k_coefficients)

    error = scaled_err.calculate_scaled_error(
        1,
        y_in,
        y_out,
        calculated_error)

    assert error == 0.0008593561322592196

    new_h = pi_step.compute_new_step_size(error, previous_error, new_h, False)

    assert new_h == 1.0516696430816863