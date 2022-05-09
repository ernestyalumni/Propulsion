from T1000.numerical.RKMethods.HermiteInterpolation import HermiteInterpolation
from collections import namedtuple

import numpy as np
import pytest

@pytest.fixture
def example_setup():

    return namedtuple(
        "ExampleSetup",
        [
            "y_0",
            "dydx_0",
            "y_1",
            "dydx_1",
            "y_2",
            "dydx_2",
            "y_3",
            "dydx_3",
            "h"])(
                np.array([0.5,]),
                np.array([1.5,]),
                np.array([1.4251302083333333,]),
                np.array([2.175644097222222,]),
                np.array([2.657246907552083,]),
                np.array([2.6408707492856616,]),
                np.array([4.0202794075012207,]),
                np.array([2.7591771182645264,]),
                0.5)


def test_HermiteInterpolationCalculatesForThetaBetween0And1(example_setup):
    setup = example_setup

    result = HermiteInterpolation.calculate_hermite_interpolation(
        setup.y_0,
        setup.y_1,
        setup.dydx_0,
        setup.dydx_1,
        0.5,
        setup.h)

    assert result[0] == 0.9203373480902778

    result = HermiteInterpolation.calculate_hermite_interpolation(
        setup.y_1,
        setup.y_2,
        setup.dydx_1,
        setup.dydx_2,
        0.5,
        setup.h)

    assert result[0] == 2.012111892188743

    result = HermiteInterpolation.calculate_hermite_interpolation(
        setup.y_2,
        setup.y_3,
        setup.dydx_2,
        setup.dydx_3,
        0.5,
        setup.h)

    assert result[0] == 3.331369009465473

def test_HermiteInterpolationCalculatesForTheta0(example_setup):
    setup = example_setup

    result = HermiteInterpolation.calculate_hermite_interpolation(
        setup.y_0,
        setup.y_1,
        setup.dydx_0,
        setup.dydx_1,
        0.0,
        setup.h)

    assert result[0] == 0.5

    result = HermiteInterpolation.calculate_hermite_interpolation(
        setup.y_1,
        setup.y_2,
        setup.dydx_1,
        setup.dydx_2,
        0.0,
        setup.h)

    assert result[0] == 1.4251302083333333

    result = HermiteInterpolation.calculate_hermite_interpolation(
        setup.y_2,
        setup.y_3,
        setup.dydx_2,
        setup.dydx_3,
        0.0,
        setup.h)

    assert result[0] == 2.657246907552083

def test_HermiteInterpolationCalculatesForTheta1(example_setup):
    setup = example_setup

    result = HermiteInterpolation.calculate_hermite_interpolation(
        setup.y_0,
        setup.y_1,
        setup.dydx_0,
        setup.dydx_1,
        1.0,
        setup.h)

    assert result[0] == 1.4251302083333333

    result = HermiteInterpolation.calculate_hermite_interpolation(
        setup.y_1,
        setup.y_2,
        setup.dydx_1,
        setup.dydx_2,
        1.0,
        setup.h)

    assert result[0] == 2.657246907552083

    result = HermiteInterpolation.calculate_hermite_interpolation(
        setup.y_2,
        setup.y_3,
        setup.dydx_2,
        setup.dydx_3,
        1.0,
        setup.h)

    assert result[0] == 4.0202794075012207
