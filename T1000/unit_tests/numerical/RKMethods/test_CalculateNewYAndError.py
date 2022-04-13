from T1000.numerical.RKMethods.CalculateNewYAndError \
    import CalculateNewYAndError
from T1000.numerical.RKMethods.DOPRI5Coefficients import DOPRI5Coefficients
from T1000.numerical.RKMethods.RK4Coefficients import RK4Coefficients
from T1000.numerical.RKMethods.kCoefficients import kCoefficients
from collections import namedtuple

import numpy as np
import pytest

@pytest.fixture
def RK4_fixture():
    calc = CalculateNewYAndError(
        RK4Coefficients.s,
        example_derivative,
        RK4Coefficients.a_coefficients,
        RK4Coefficients.c_coefficients,
        RK4Coefficients.delta_coefficients)

    return calc


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

    return namedtuple("ExampleSetup", ["x_0", "y_0", "dydx_0", "h"])(
        0.0,
        np.array([0.5,]),
        np.array([1.5,]),
        0.5)


def example_derivative(t, y):
    """
    @param k [in/out] k serves as the output.
    """
    return y - t * t + 1.0

def example_exact_solution(t):
    return t * t + 2 * t + 1 - 0.5 * np.exp(t)


def test_CalculateNewYAndErrorConstructsWithRK4Coefficients():

    calc = CalculateNewYAndError(
        RK4Coefficients.s,
        example_derivative,
        RK4Coefficients.a_coefficients,
        RK4Coefficients.c_coefficients,
        RK4Coefficients.delta_coefficients)

    assert True

def test_steps_for_sum_a_and_k_products_with_RK4Coefficients(
        RK4_fixture,
        example_setup):
    calc = RK4_fixture

    k_coefficients = kCoefficients(4, 1)
    assert k_coefficients.size() == 4
    assert k_coefficients.dimensions() == (1,)

    assert calc.get_a_ij(2, 1) == 0.5

    k_coefficients._k_coefficients[0] = example_setup.dydx_0
    assert k_coefficients.get_ith_coefficient(1) == np.array([1.5,])

    a_lj_times_kj = k_coefficients.scalar_multiply(1, calc.get_a_ij(2, 1))

    assert a_lj_times_kj == np.array([0.75,])


def test_calculate_sum_a_and_k_products_with_RK4Coefficients(
        RK4_fixture,
        example_setup):
    calc = RK4_fixture
    setup = example_setup

    k_coefficients = kCoefficients(4, 1)

    k_coefficients._k_coefficients[0] = setup.dydx_0

    assert len(k_coefficients._k_coefficients) == 4
    assert k_coefficients._k_coefficients[0][0] == 1.5

    x_l = setup.x_0 + calc.get_c_i(2) * setup.h

    assert x_l == 0.25

    y_out = None

    y_out = calc._sum_a_and_k_products(k_coefficients, 2, setup.h)
    assert k_coefficients._k_coefficients[0][0] == 1.5
    assert y_out == np.array([0.375,])

def test_steps_for_apply_method_with_RK4Coefficients(
        RK4_fixture,
        example_setup):
    calc = RK4_fixture
    setup = example_setup
    k_coefficients = kCoefficients(4, 1)

    k_coefficients._k_coefficients[0] = setup.dydx_0

    x_l = setup.x_0 + calc.get_c_i(2) * setup.h
    assert x_l == 0.25

    y_out = None

    y_out = calc._sum_a_and_k_products(k_coefficients, 2, setup.h)

    y_out += setup.y_0

    assert y_out == np.array([0.875,])

    k_coefficients._k_coefficients[2 - 1] = example_derivative(x_l, y_out)

    assert k_coefficients._k_coefficients[2 - 1] == np.array([1.8125,])


def test_apply_method_works_with_RK4Coefficients(RK4_fixture, example_setup):
    calc = RK4_fixture
    setup = example_setup
    k_coefficients = kCoefficients(RK4Coefficients.s, 1)
    y_out = np.array([0.0,])

    # Step 1.

    y_out = calc.apply_method(
        setup.h,
        setup.x_0,
        setup.y_0,
        setup.dydx_0,
        k_coefficients)

    assert k_coefficients._k_coefficients[0] == np.array([1.5,])
    assert k_coefficients._k_coefficients[1] == np.array([1.8125,])
    assert k_coefficients._k_coefficients[2] == np.array([1.890625,])
    assert k_coefficients._k_coefficients[3] == np.array([2.1953125,])

    assert y_out == np.array([1.4453125,])

    y_out = setup.y_0

    for i in range(RK4Coefficients.s):

        y_out += setup.h * (
            float(RK4Coefficients.b_coefficients.get_ith_element(i + 1)) *
                k_coefficients.get_ith_coefficient(i + 1))

    assert y_out[0] == pytest.approx(1.425130210833333)

    # Step 2

    y_in = y_out
    y_out = calc.apply_method(
        setup.h,
        setup.x_0 + setup.h,
        y_out,
        example_derivative(setup.x_0 + setup.h, y_out),
        k_coefficients)

    assert k_coefficients._k_coefficients[0][0] == pytest.approx(
        2.175130208333334)
    assert k_coefficients._k_coefficients[1][0] == pytest.approx(
        2.406412760416666)
    assert k_coefficients._k_coefficients[2][0] == pytest.approx(
        2.4642333984375)
    assert k_coefficients._k_coefficients[3][0] == pytest.approx(
        2.657246907552084)
    assert y_out.size == 1
    assert y_out[0] == pytest.approx(2.657246907552083)

    assert y_in[0] == pytest.approx(1.425130210833333)
    y_out = y_in

    for i in range(RK4Coefficients.s):

        y_out += setup.h * (
            float(RK4Coefficients.b_coefficients.get_ith_element(i + 1)) *
                k_coefficients.get_ith_coefficient(i + 1))

    assert y_out[0] == pytest.approx(2.639602661132812)

    # Step 3

    y_in = y_out
    y_out = calc.apply_method(
        setup.h,
        setup.x_0 + 2 * setup.h,
        y_out,
        example_derivative(setup.x_0 + 2 * setup.h, y_out),
        k_coefficients)

    assert k_coefficients._k_coefficients[0][0] == pytest.approx(
        2.639602661132812)
    assert k_coefficients._k_coefficients[1][0] == pytest.approx(
        2.737003326416016)
    assert k_coefficients._k_coefficients[2][0] == pytest.approx(
        2.761353492736816)
    assert k_coefficients._k_coefficients[3][0] == pytest.approx(
        2.77027940750122)

    y_out = y_in

    for i in range(RK4Coefficients.s):

        y_out += setup.h * (
            float(RK4Coefficients.b_coefficients.get_ith_element(i + 1)) *
                k_coefficients.get_ith_coefficient(i + 1))

    assert y_out[0] == pytest.approx(4.006818970044454)


def test_apply_method_works_with_DOPRI5Coefficients(
        DOPRI5_fixture,
        example_setup):
    calc = DOPRI5_fixture
    setup = example_setup
    k_coefficients = kCoefficients(DOPRI5Coefficients.s, 1)
    y_out = np.array([0.0,])

    # Step 1.

    assert setup.h == 0.5
    assert setup.x_0 == 0.0
    assert setup.y_0[0] == 0.5
    assert setup.dydx_0[0] == 1.5

    y_out = calc.apply_method(
        setup.h,
        setup.x_0,
        setup.y_0,
        setup.dydx_0,
        k_coefficients)

    assert k_coefficients._k_coefficients[0] == np.array([1.5,])
    assert k_coefficients._k_coefficients[1][0] == pytest.approx(
        1.6400000000000001)
    assert k_coefficients._k_coefficients[2][0] == pytest.approx(1.71825)
    assert k_coefficients._k_coefficients[3][0] == pytest.approx(
        2.066666666666666)
    assert k_coefficients._k_coefficients[4][0] == pytest.approx(
        2.1469574759945127)
    assert k_coefficients._k_coefficients[5][0] == pytest.approx(
        2.2092840909090903)
    assert k_coefficients._k_coefficients[6][0] == pytest.approx(
        2.175644097222222)

    assert y_out[0] == pytest.approx(
        example_exact_solution(setup.h),
        0.00001)

    # Step 2

    y_in = y_out
    y_out = calc.apply_method(
        setup.h,
        setup.x_0 + setup.h,
        y_out,
        example_derivative(setup.x_0 + setup.h, y_out),
        k_coefficients)

    assert k_coefficients._k_coefficients[0][0] == pytest.approx(
        2.175644097222222)
    assert k_coefficients._k_coefficients[1][0] == pytest.approx(
        2.2832085069444443)
    assert k_coefficients._k_coefficients[2][0] == pytest.approx(
        2.341591707899305)
    assert k_coefficients._k_coefficients[3][0] == pytest.approx(
        2.5801328124999987)
    assert k_coefficients._k_coefficients[4][0] == pytest.approx(
        2.633202642222983)
    assert k_coefficients._k_coefficients[5][0] == pytest.approx(
        2.667628162582859)
    assert k_coefficients._k_coefficients[6][0] == pytest.approx(
        2.6408707492856616)

    assert y_out[0] == pytest.approx(
        example_exact_solution(2 * setup.h),
        0.00001)

    y_in = y_out
    y_out = calc.apply_method(
        setup.h,
        setup.x_0 + 2 * setup.h,
        y_out,
        example_derivative(setup.x_0 + 2 * setup.h, y_out),
        k_coefficients)

    assert k_coefficients._k_coefficients[0][0] == pytest.approx(
        2.6408707492856616)
    assert k_coefficients._k_coefficients[1][0] == pytest.approx(
        2.694957824214227)
    assert k_coefficients._k_coefficients[2][0] == pytest.approx(
        2.7205861576079746)
    assert k_coefficients._k_coefficients[3][0] == pytest.approx(
        2.77797279059516)
    assert k_coefficients._k_coefficients[4][0] == pytest.approx(
        2.786162739074309)
    assert k_coefficients._k_coefficients[5][0] == pytest.approx(
        2.7745870563781194)
    assert k_coefficients._k_coefficients[6][0] == pytest.approx(
        2.7591771182645264)

    assert y_out[0] == pytest.approx(
        example_exact_solution(3 * setup.h),
        0.00001)
