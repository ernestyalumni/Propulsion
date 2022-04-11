from T1000.numerical.RKMethods.CalculateNewYAndError import CalculateNewYAndError
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


def example_derivative(t, y):
    """
    @param k [in/out] k serves as the output.
    """
    return y - t * t + 1.0

example_setup = namedtuple("ExampleSetup", ["x_0", "y_0", "dydx_0", "h"])(
    0.0,
    np.array([0.5,]),
    np.array([1.5,]),
    0.5)


def test_CalculateNewYAndErrorConstructsWithRK4Coefficients():

    calc = CalculateNewYAndError(
        RK4Coefficients.s,
        example_derivative,
        RK4Coefficients.a_coefficients,
        RK4Coefficients.c_coefficients,
        RK4Coefficients.delta_coefficients)

    assert True

def test_steps_for_sum_a_and_k_products_with_RK4Coefficients(RK4_fixture):
    calc = RK4_fixture

    k_coefficients = kCoefficients(4, 1)
    assert k_coefficients.size() == 4
    assert k_coefficients.dimensions() == (1,)

    assert calc.get_a_ij(2, 1) == 0.5

    k_coefficients._k_coefficients[0] = example_setup.dydx_0
    assert k_coefficients.get_ith_coefficient(1) == np.array([1.5,])

    a_lj_times_kj = k_coefficients.scalar_multiply(1, calc.get_a_ij(2, 1))

    assert a_lj_times_kj == np.array([0.75,])


def test_calculate_sum_a_and_k_products_with_RK4Coefficients(RK4_fixture):
    calc = RK4_fixture

    k_coefficients = kCoefficients(4, 1)

    k_coefficients._k_coefficients[0] = example_setup.dydx_0

    assert len(k_coefficients._k_coefficients) == 4
    assert k_coefficients._k_coefficients[0][0] == 1.5

    x_l = example_setup.x_0 + calc.get_c_i(2) * example_setup.h

    assert x_l == 0.25

    y_out = None

    y_out = calc._sum_a_and_k_products(k_coefficients, 2, example_setup.h)
    assert k_coefficients._k_coefficients[0][0] == 1.5
    assert y_out == np.array([0.375,])

def test_steps_for_apply_method_with_RK4Coefficients(RK4_fixture):
    calc = RK4_fixture
    k_coefficients = kCoefficients(4, 1)

    k_coefficients._k_coefficients[0] = example_setup.dydx_0

    x_l = example_setup.x_0 + calc.get_c_i(2) * example_setup.h
    assert x_l == 0.25

    y_out = None

    y_out = calc._sum_a_and_k_products(k_coefficients, 2, example_setup.h)

    y_out += example_setup.y_0

    assert y_out == np.array([0.875,])

    k_coefficients._k_coefficients[2 - 1] = example_derivative(x_l, y_out)

    assert k_coefficients._k_coefficients[2 - 1] == np.array([1.8125,])


def test_apply_method_works_with_RK4Coefficients(RK4_fixture):
    calc = RK4_fixture
    k_coefficients = kCoefficients(4, 1)
    y_out = np.array([0.0,])

    # Step 1.

    y_out = calc.apply_method(
        example_setup.h,
        example_setup.x_0,
        example_setup.y_0,
        example_setup.dydx_0,
        k_coefficients)

    assert k_coefficients._k_coefficients[0] == np.array([1.5,])
    assert k_coefficients._k_coefficients[1] == np.array([1.8125,])
    assert k_coefficients._k_coefficients[2] == np.array([1.890625,])
    assert k_coefficients._k_coefficients[3] == np.array([2.1953125,])

    assert y_out == np.array([1.4453125,])

    y_out = example_setup.y_0

    for i in range(RK4Coefficients.s):

        y_out += example_setup.h * (
            float(RK4Coefficients.b_coefficients.get_ith_element(i + 1)) *
                k_coefficients.get_ith_coefficient(i + 1))

    assert y_out[0] == pytest.approx(1.425130210833333)

    # Step 2

    y_in = y_out
    y_out = calc.apply_method(
        example_setup.h,
        example_setup.x_0 + example_setup.h,
        y_out,
        example_derivative(example_setup.x_0 + example_setup.h, y_out),
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

        y_out += example_setup.h * (
            float(RK4Coefficients.b_coefficients.get_ith_element(i + 1)) *
                k_coefficients.get_ith_coefficient(i + 1))

    assert y_out[0] == pytest.approx(2.639602661132812)

    y_in = y_out
    y_out = calc.apply_method(
        example_setup.h,
        example_setup.x_0 + 2 * example_setup.h,
        y_out,
        example_derivative(example_setup.x_0 + 2 * example_setup.h, y_out),
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

        y_out += example_setup.h * (
            float(RK4Coefficients.b_coefficients.get_ith_element(i + 1)) *
                k_coefficients.get_ith_coefficient(i + 1))

    assert y_out[0] == pytest.approx(4.006818970044454)
