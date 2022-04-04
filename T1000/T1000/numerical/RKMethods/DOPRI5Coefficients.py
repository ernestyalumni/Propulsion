from .aCoefficients import aCoefficients
from .bCoefficients import bCoefficients
from sympy import Rational

class DOPRI5Coefficients:

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
