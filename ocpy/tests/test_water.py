import numpy as np
import pytest

from ocpy.water.scattering import PMH, betasw_ZHH2009, dlnasw_ds, rhou_sw


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
    # Tc = 25, S = 35. This test used to pin -1.56e16, the value of the
    # transcription error (two T^3 coefficients at 1e+11 instead of 1e-11),
    # so it protected the bug. Millero & Leung (1976) give a small negative
    # derivative of order 1e-4.
    Tc = 25
    S = 35
    np.testing.assert_allclose(dlnasw_ds(Tc, S), -5.7223347624801e-4, rtol=1e-9)
    # ... and EPFT-UP's independent port agrees at 20 degC to every digit
    np.testing.assert_allclose(dlnasw_ds(20, 35), -5.709442270740529e-4,
                               rtol=1e-12)
    assert -1e-3 < dlnasw_ds(Tc, S) < 0


def test_betasw_ZHH2009():
    """Zhang, Hu & He (2009) pure-seawater scattering, now usable.

    Pins b_sw for seawater (20 degC, S = 35) against the value the corrected
    formulas give -- identical to EPFT-UP's independent port -- and checks the
    physics: molecular lambda^-4.2 to -4.3 slope, roughly 30% salt
    enhancement over pure water, and the 1 + f cos^2 angular shape.
    """
    w = np.array([400., 500., 600.])
    betasw, beta90, bsw = betasw_ZHH2009(w, 20., np.array([0., 90., 180.]), 35.)
    bsw = np.ravel(bsw)
    np.testing.assert_allclose(bsw[1], 2.54734e-3, rtol=1e-4)
    assert 2.4e-3 < bsw[1] < 2.7e-3
    slope = np.log(bsw[0] / bsw[2]) / np.log(400 / 600)
    assert -4.35 < slope < -4.15
    b0 = np.ravel(betasw_ZHH2009(w, 20., 90., 0.)[2])     # scalar theta is fine
    assert np.all((bsw / b0 > 1.25) & (bsw / b0 < 1.35))
    assert betasw.shape == (3, 3)
    np.testing.assert_allclose(betasw[0], betasw[2])      # symmetric fore/aft
    np.testing.assert_allclose(betasw[1], beta90)


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
