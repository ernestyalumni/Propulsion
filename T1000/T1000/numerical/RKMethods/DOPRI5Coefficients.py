from .aCoefficients import aCoefficients
from .bCoefficients import bCoefficients
from .cCoefficients import cCoefficients
from sympy import Rational

class DOPRI5Coefficients:

    s = 7

    a_coefficients = aCoefficients(DOPRI5Coefficients::s, [
        0.2,
        3.0 / 40.0,
        9.0 / 40.0,
        44.0 / 45.0,
        -56.0 / 15.0,
        32.0 / 9.0,
        19372.0 / 6561.0,
        -25360.0 / 2187.0,
        64448.0 / 6561.0,
        -212.0 / 729.0,
        9017.0 / 3168.0,
        -355.0 / 33.0,
        46732.0 / 5247.0,
        49.0 / 176.0,
        -5103.0 / 18656.0,
        35.0 / 384.0,
        0.0,
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0])

    b_coefficients = bCoefficients(7, [
        Rational(35, 384),
        0,
        Rational(500, 1113),
        Rational(125, 192),
        Rational(-2187, 6784),
        Rational(11, 84),
        0])

    bstar_coefficients = bCoefficients(7, [
        Rational(5179, 57600),
        0,
        Rational(7571, 16695),
        Rational(393, 640),
        Rational(-92097, 339200),
        Rational(187, 2100),
        Rational(1, 40)])

    c_coefficients = cCoefficients(7, [
        71.0 / 57600.0,
        0.0,
        -71.0 / 16695.0,
        71.0 / 1920.0,
        -17253.0 / 339200.0,
        22.0 / 525.0,
        -1.0 / 40.0])
