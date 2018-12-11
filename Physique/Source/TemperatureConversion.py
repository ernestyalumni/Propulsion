"""
@name TemperatureConversion.py
@file TemperatureConversion.py
@author Ernest Yeung
@date 20150913
@email ernestyalumni@gmail.com
@brief I implement temperature conversion with symbolic computation in sympy
@ref
@details
@copyright If you find this code useful, feel free to donate directly and easily
at this direct PayPal link: 

https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 

which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.
Otherwise, I receive emails and messages on how all my (free) material on
physics, math, and engineering have helped students with their studies, and I
know what it's like to not have money as a student, but love physics (or math,
sciences, etc.), so I am committed to keeping all my material open-source and
free, whether or not sufficiently crowdfunded, under the open-source MIT
license: feel free to copy, edit, paste, make your own versions, share, use as
you wish.    

Peace out, never give up! -EY
"""
import sympy
from sympy import Eq
from sympy import Rational as Rat
from sympy import symbols
from sympy.solvers import solve

T_F, T_C, T_K = symbols("T_F T_C T_K", real=True)

FahrenheitCelsiusConversion = Eq(T_F, T_C * (Rat(9) / Rat(5)) + Rat(32)) 

KelvinCelsiusConversion = Eq(T_K, T_C + 273.15)
