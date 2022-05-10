from T1000.numerical.RKMethods.CalculateDenseOutputCoefficient \
    import CalculateDenseOutputCoefficient
from T1000.numerical.RKMethods.DOPRI5Coefficients import DOPRI5Coefficients
from T1000.numerical.RKMethods.HermiteInterpolation import HermiteInterpolation
from T1000.numerical.RKMethods.kCoefficients import kCoefficients
from unit_tests.numerical.RKMethods.fixtures import DOPRI5_fixture, example_setup
from collections import namedtuple

import numpy as np
import pytest

@pytest.fixture
def calculate_dense_output_fixture():
    calc_dense = CalculateDenseOutputCoefficient(
        DOPRI5Coefficients.s,
        DOPRI5Coefficients.dense_output_coefficients)
    return calc_dense

# cf. pp. 918-919 Numerical Recipes, void StepperDopr5<D>::prepare_dense(...)
class PrepareDense:
    def __init__(self, dense_output_coefficients):
        self._dense_output_coefficients = dense_output_coefficients
    
    def prepare_dense(self, y_0, y_1, dydx_0, dydx_1, k_coefficients, h):
        self.rcont1 = y_0
        ydiff = y_1 - y_0
        self.rcont2 = ydiff
        bsp1 = h * dydx_0 - ydiff
        self.rcont3 = bsp1
        self.rcont4 = ydiff - h * dydx_1 - bsp1
        self.rcont5 = h * (
            self._dense_output_coefficients.get_ith_element(1) * dydx_0 + \
            self._dense_output_coefficients.get_ith_element(3) * \
                k_coefficients._k_coefficients[2] + \
            self._dense_output_coefficients.get_ith_element(4) * \
                k_coefficients._k_coefficients[3] + \
            self._dense_output_coefficients.get_ith_element(5) * \
                k_coefficients._k_coefficients[4] + \
            self._dense_output_coefficients.get_ith_element(6) * \
                k_coefficients._k_coefficients[5] + \
            self._dense_output_coefficients.get_ith_element(7) * dydx_1)

    def dense_out(self, theta, h):
        """
        @ref pp. 919 17.2 Adaptive Stepsize Control for Runge-Kutta, Numerical
        Recipes.
        """
        theta1 = 1.0 - theta
        return self.rcont1 + theta * (self.rcont2 + theta1 * (
            self.rcont3 + theta * (self.rcont4 + theta1 * self.rcont5)))


@pytest.fixture
def prepare_dense_dense_out_fixture():
    prepare_dense = PrepareDense(DOPRI5Coefficients.dense_output_coefficients)
    return prepare_dense


def test_calculate_dense_output_coefficient_constructs():
    calc_dense = CalculateDenseOutputCoefficient(
        DOPRI5Coefficients.s,
        DOPRI5Coefficients.dense_output_coefficients)

    assert True


def test_calculate_dense_output_coefficient_calculates(
        DOPRI5_fixture,
        example_setup,
        calculate_dense_output_fixture,
        prepare_dense_dense_out_fixture):
    calc = DOPRI5_fixture
    setup = example_setup
    calc_dense = calculate_dense_output_fixture
    prepare_dense = prepare_dense_dense_out_fixture
    k_coefficients = kCoefficients(DOPRI5Coefficients.s, 1)
    y_out = np.array([0.0,])

    # Step 1.

    y_out = calc.apply_method(
        setup.h,
        setup.x_0,
        setup.y_0,
        setup.dydx_0,
        k_coefficients)

    interpolation = HermiteInterpolation.calculate_hermite_interpolation(
        setup.y_0,
        y_out,
        setup.dydx_0,
        k_coefficients._k_coefficients[6],
        0.5,
        setup.h)

    assert interpolation[0] == 0.9205942925347221

    result = calc_dense.calculate_dense_output_coefficient(
        k_coefficients,
        0.5,
        setup.h)

    assert result[0] == pytest.approx(-9.84422411898500e-5)

    prepare_dense.prepare_dense(
        setup.y_0,
        y_out,
        setup.dydx_0,
        k_coefficients._k_coefficients[6],
        k_coefficients,
        setup.h)

    expected_output = prepare_dense.dense_out(0.5, setup.h)

    assert (interpolation + result)[0] == expected_output

    # Step 2

    y_in = y_out
    dydx_0 = k_coefficients._k_coefficients[6]
    y_out = calc.apply_method(
        setup.h,
        setup.x_0 + setup.h,
        y_in,
        dydx_0,
        k_coefficients)

    assert y_out[0] == pytest.approx(2.6408590857704777, 1.0e-5)

    interpolation = HermiteInterpolation.calculate_hermite_interpolation(
        y_in,
        y_out,
        dydx_0,
        k_coefficients._k_coefficients[6],
        0.5,
        setup.h)

    assert interpolation[0] == 2.004180757499977

    result = calc_dense.calculate_dense_output_coefficient(
        k_coefficients,
        0.5,
        setup.h)

    assert result[0] == pytest.approx(-0.000167097453235482)

    prepare_dense.prepare_dense(
        y_in,
        y_out,
        dydx_0,
        k_coefficients._k_coefficients[6],
        k_coefficients,
        setup.h)

    expected_output = prepare_dense.dense_out(0.5, setup.h)

    assert (interpolation + result)[0] == expected_output[0]

    # Step 3

    y_in = y_out
    dydx_0 = k_coefficients._k_coefficients[6]
    y_out = calc.apply_method(
        setup.h,
        setup.x_0 + 2 * setup.h,
        y_in,
        dydx_0,
        k_coefficients)

    assert y_out[0] == pytest.approx(4.009155464830968, 1.0e-5)

    interpolation = HermiteInterpolation.calculate_hermite_interpolation(
        y_in,
        y_out,
        dydx_0,
        k_coefficients._k_coefficients[6],
        0.5,
        setup.h)

    assert interpolation[0] == 3.317629785713915

    result = calc_dense.calculate_dense_output_coefficient(
        k_coefficients,
        0.5,
        setup.h)

    assert result[0] == pytest.approx(-0.000280290946199302)

    prepare_dense.prepare_dense(
        y_in,
        y_out,
        dydx_0,
        k_coefficients._k_coefficients[6],
        k_coefficients,
        setup.h)

    expected_output = prepare_dense.dense_out(0.5, setup.h)

    assert (interpolation + result)[0] == expected_output[0]
