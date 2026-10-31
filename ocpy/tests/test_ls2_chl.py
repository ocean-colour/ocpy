"""Tests for LS2's chlorophyll side-chain: ``bp_from_chla`` and the OC4 forms.

The oracle is the ``Input bp`` column of ``files/LS2_test_run.csv``, the
authors' own reference vector.  It carries no chlorophyll, so the amplitude of
``b_p(660) = 0.347 Chl^0.766`` cannot be checked against it independently:
the implied Chl is recovered by inverting that relation.  What the column *does*
pin is the spectral shape, which is an exact ``lambda^-1`` law to 2e-15.

The implied Chl is close to OC4v4 of the same row's Rrs but not equal to it
(-2% to +14%), so the authors' Chl came from elsewhere -- presumably measured
-- and OC4v4 cannot serve as an end-to-end oracle here either.  That is pinned
too, so a future reader does not go looking for an exact match.
"""

import pathlib

import numpy as np
import pandas
import pytest

from ocpy.chl.band_ratios import OC4V4_COEFFS, oc4, oc4v4
from ocpy.iop.scattering import bp_from_chla

WAVES = np.array((412., 443., 490., 510., 555., 670.))


@pytest.fixture(scope='module')
def reference():
    """``(Rrs, bp)`` from the authors' reference vector, each ``(10, 6)``."""
    path = pathlib.Path(__file__).parent / 'files' / 'LS2_test_run.csv'
    df = pandas.read_csv(path)
    blocks = [df[df['Input wavelength [nm]'] == w].reset_index(drop=True)
              for w in WAVES]

    def _col(name):
        return np.column_stack([b[name].values for b in blocks])

    return _col('Input Rrs [1/sr]'), _col('Input bp [1/m]')


def _implied_chl(bp):
    """Invert ``b_p(lambda) = 0.347 Chl^0.766 (lambda/660)^-1`` row by row."""
    bp660 = np.mean(bp * WAVES / 660., axis=1)
    return (bp660 / 0.347) ** (1. / 0.766)


def test_reference_bp_is_an_exact_inverse_power_law(reference):
    """The authors' ``b_p`` is ``lambda^-1``, not MM01's Chl-dependent slope."""
    _, bp = reference
    flat = bp * WAVES
    spread = flat.std(axis=1) / flat.mean(axis=1)
    assert spread.max() < 1e-12


def test_bp_from_chla_reproduces_reference(reference):
    """From its implied Chl, every reference ``b_p`` comes back to 1e-12."""
    _, bp = reference
    chl = _implied_chl(bp)
    np.testing.assert_allclose(bp_from_chla(WAVES, chl), bp, rtol=1e-12)


def test_bp_from_chla_values():
    """The published relation at round numbers, independent of the CSV."""
    np.testing.assert_allclose(bp_from_chla(660., 1.), 0.347, rtol=1e-15)
    np.testing.assert_allclose(bp_from_chla(330., 1.), 0.694, rtol=1e-15)
    np.testing.assert_allclose(bp_from_chla(660., 10.), 0.347 * 10 ** 0.766,
                               rtol=1e-15)


def test_bp_from_chla_vectorization():
    """``(N,)`` Chl x ``(L,)`` wave gives ``(N, L)``, equal to a scalar loop."""
    rng = np.random.default_rng(2)
    chl = 10 ** rng.uniform(-2, 1.5, 7)
    wave = np.linspace(400., 750., 71)

    block = bp_from_chla(wave, chl)
    assert block.shape == (7, 71)
    assert bp_from_chla(wave, chl[0]).shape == (71,)
    assert np.ndim(bp_from_chla(500., 0.3)) == 0

    loop = np.array([[float(bp_from_chla(w, c)) for w in wave] for c in chl])
    np.testing.assert_allclose(block, loop, rtol=1e-15)


def test_bp_from_chla_bad_chl_is_nan(recwarn):
    """Negative or non-finite Chl gives NaN, without a RuntimeWarning."""
    out = bp_from_chla(WAVES, np.array([-0.1, np.nan, np.inf, 0., 1.]))
    assert np.all(np.isnan(out[:3]))
    assert np.all(out[3] == 0.)
    assert np.all(np.isfinite(out[4]))
    assert len(recwarn) == 0, [str(w.message) for w in recwarn]


def test_oc4v4_regression(reference):
    """OC4v4 on the reference Rrs, pinned, and checked against the formula.

    The pinned values guard the coefficients; the explicit polynomial
    guards the band selection and the absence of an additive term.
    """
    Rrs, _ = reference
    chl = oc4v4(WAVES, Rrs)
    assert chl.shape == (10,)

    R = np.log10(np.max(Rrs[:, 1:4], axis=1) / Rrs[:, 4])
    poly = 10 ** sum(c * R ** k for k, c in enumerate(OC4V4_COEFFS))
    np.testing.assert_allclose(chl, poly, rtol=1e-14)

    np.testing.assert_allclose(
        chl[[0, 3, 6, 8]],
        [0.57430452, 0.39480105, 0.10817861, 0.09077881], rtol=1e-7)

    # Vectorized == one spectrum at a time.
    loop = np.array([oc4v4(WAVES, r) for r in Rrs])
    np.testing.assert_allclose(chl, loop, rtol=1e-15)


def test_oc4_1998_regression(reference):
    """The 1998 modified-cubic OC4, pinned so it cannot drift.

    BING and IOPtics call :func:`oc4`, so it must not change when
    :func:`oc4v4` is added beside it.
    """
    Rrs, _ = reference
    chl = np.array([oc4(WAVES, r) for r in Rrs])
    np.testing.assert_allclose(
        chl[[0, 3, 6, 8]],
        [0.57375457, 0.38745994, 0.10718694, 0.08965597], rtol=1e-7)

    # The two forms are close but distinct on these spectra.
    rel = chl / oc4v4(WAVES, Rrs) - 1.
    assert np.all(np.abs(rel) < 0.03)
    assert np.all(rel < 0.)


def test_oc4v4_band_guard():
    """A missing OC4 band raises rather than silently using a far one."""
    wave = np.array([412., 443., 490., 555., 670.])   # no 510
    with pytest.raises(ValueError, match='510'):
        oc4v4(wave, np.ones(5) * 1e-3)
    # Hyperspectral grids are fine.
    hyper = np.arange(400., 751., 5.)
    assert np.isfinite(oc4v4(hyper, np.full(hyper.size, 2e-3)))


def test_oc4v4_bad_ratio_is_nan(recwarn):
    """A non-positive or non-finite band ratio gives NaN, quietly."""
    Rrs = np.array([[1e-3, 1e-3, 1e-3, 1e-3, 0.],
                    [-1e-3, -1e-3, -1e-3, 1e-3, 1e-3],
                    [1e-3, 1e-3, 1e-3, 1e-3, 1e-3]])
    wave = np.array([443., 490., 510., 555., 670.])
    Rrs[0, 3] = 0.
    out = oc4v4(wave, Rrs)
    assert np.isnan(out[0]) and np.isnan(out[1])
    np.testing.assert_allclose(out[2], 10 ** OC4V4_COEFFS[0], rtol=1e-15)
    assert len(recwarn) == 0, [str(w.message) for w in recwarn]


def test_implied_chl_is_not_oc4v4(reference):
    """The reference Chl was not derived from these Rrs by OC4v4.

    Measured: OC4v4 / implied Chl spans 0.975-1.140.  Close enough to say
    the authors' Chl is plausible for these waters; far enough to say it is
    an independent input, so the CSV cannot test the amplitude 0.347.
    """
    Rrs, bp = reference
    ratio = oc4v4(WAVES, Rrs) / _implied_chl(bp)
    assert 0.95 < ratio.min() < 0.99
    assert 1.10 < ratio.max() < 1.20
