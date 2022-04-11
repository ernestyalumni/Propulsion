from .aCoefficients import aCoefficients
from .bCoefficients import bCoefficients
from .cCoefficients import cCoefficients
from sympy import Rational

class RK4Coefficients:

    s = 4

    a_coefficients = aCoefficients(4, [
        0.5,
        0.0,
        0.5,
        0.0,
        0.0,
        1.0
        ])

    a_extended_coefficients = aCoefficients(5, [
        0.5,
        0.0,
        0.5,
        0.0,
        0.0,
        1.0,
        1.0 / 6.0,
        2.0 / 6.0,
        2.0 / 6.0,
        1.0 / 6.0
        ])


    b_coefficients = bCoefficients(4, [
        Rational(1, 6),
        Rational(2, 6),
        Rational(2, 6),
        Rational(1, 6)])

    c_coefficients = cCoefficients(4, [
        0.5,
        0.5,
        1.0])

    delta_coefficients = bCoefficients(5, [
        Rational(1, 6) - Rational(-1, 2),
        Rational(1, 3) - Rational(7, 3),
        Rational(1, 3) - Rational(7, 3),
        Rational(1, 6) - Rational(13, 6),
        0 - Rational(-16, 3)])