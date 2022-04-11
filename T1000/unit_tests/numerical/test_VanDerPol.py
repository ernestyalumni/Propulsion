from T1000.numerical.VanDerPol import van_der_pol

import pytest

def test_trivial_multiprocess_example():

    assert van_der_pol(0.0, (0.0, 1.5), 0.5) == [1.5, 0.75]

    return
