from oceancolor.water.scattering import PMH, dlnasw_ds, rhou_sw

import pytest


def test_PMH():
    # Test case 1: n_wat = 1.33
    n_wat = 1.33
    expected_result = 0.8406442577397122
    assert PMH(n_wat) == expected_result

    # Test case 2: n_wat = 1.34
    n_wat = 1.34
    expected_result = 0.8744535765052853
    assert PMH(n_wat) == expected_result

    # Test case 3: n_wat = 1.35
    n_wat = 1.35
    expected_result = 0.9089477926399053
    assert PMH(n_wat) == expected_result

def test_dlnasw_ds():
    # Test case 1: Tc = 25, S = 35
    Tc = 25
    S = 35
    expected_result = -1.5631763548946316e+16
    assert dlnasw_ds(Tc, S) == expected_result


def test_rhou_sw():
    # Test case 1: Tc = 25, S = 35
    Tc = 25
    S = 35
    expected_result = 1023.3430584772268
    assert rhou_sw(Tc, S) == expected_result

    # Test case 2: Tc = 20, S = 30
    Tc = 20
    S = 30
    expected_result = 1020.9538750640842
    assert rhou_sw(Tc, S) == expected_result
