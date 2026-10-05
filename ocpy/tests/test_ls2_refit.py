"""Tests for the LS2 coefficient refit (``ocpy.ls2.refit``).

A synthetic corpus generated from known smooth coefficients pins the fit (it
must recover them), the table export (a drop-in ``LS2_LUT`` that
``ls2_invert`` accepts and that agrees with the smooth model at the nodes) and
the file round trip.  The shipped L23 table is checked to load and run.
"""

import os
from importlib import resources

import numpy as np
import pytest

from ocpy.ls2 import ls2_main, refit


def _truth_model():
    """Smooth coefficients of roughly the published magnitudes."""
    rng = np.random.default_rng(1)
    beta_a = np.zeros((4, 3, 2))
    beta_a[0, 0] = [0.05, 0.98]          # c0 ~ 0.05 + 0.98/mu_w
    beta_a[1, :, 0] = [60., -20., 5.]
    beta_a[2, :, 0] = [-1500., 400., 100.]
    beta_a[3, :, 0] = [20000., 5000., -3000.]
    beta_b = np.zeros((3, 3, 2))
    beta_b[0, :, 1] = [18., -2., 1.]     # d0 ~ mu_w (...)
    beta_b[1, :, 1] = [-600., 100., 50.]
    beta_b[2, :, 1] = [10000., 2000., -500.]
    beta_a[1:, :, 1] += rng.normal(scale=1e-3, size=(3, 3))
    return refit.SmoothCoefficients(beta_a, beta_b, 2, ('inv', 'lin'))


def _corpus(model, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    eta = rng.uniform(0.001, 0.2, n)
    muw = rng.choice(np.cos(np.arcsin(np.sin(np.deg2rad([0, 30, 60])) / 1.34)), n)
    rrs = rng.uniform(5e-4, 8e-3, n)
    kd = rng.uniform(0.02, 0.5, n)
    a, bb = model.invert(rrs, kd, eta, muw)
    return rrs, kd, a, bb, eta, muw


def test_fit_recovers_known_coefficients():
    true = _truth_model()
    rrs, kd, a, bb, eta, muw = _corpus(true)
    m = refit.fit(rrs, kd, a, bb, eta, muw, eta_degree=2, muw_basis=('inv', 'lin'))
    np.testing.assert_allclose(m.beta_a, true.beta_a, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(m.beta_b, true.beta_b, rtol=1e-6, atol=1e-6)
    assert m.muw_basis == ('inv', 'lin') and m.domain['n_cells'] == rrs.size


def test_one_basis_for_both_tables_and_nonfinite_cells_dropped():
    true = _truth_model()
    rrs, kd, a, bb, eta, muw = _corpus(true, n=500)
    a = a.copy()
    a[:10] = np.nan
    m = refit.fit(rrs, kd, a, bb, eta, muw, muw_basis='lin')
    assert m.muw_basis == ('lin', 'lin') and m.domain['n_cells'] == 490


def test_to_lut_is_a_drop_in_table():
    lut_path = os.path.join(resources.files('ocpy'), 'data', 'LS2', 'LS2_LUT.npz')
    published = dict(np.load(lut_path))
    m = _truth_model()
    lut = m.to_lut(published)
    assert lut['a'].shape == (21, 8, 4) and lut['bb'].shape == (21, 8, 3)
    np.testing.assert_array_equal(lut['kappa'], published['kappa'])
    # at a node, the table and the smooth model agree exactly
    eta_n = published['eta'].ravel()[10]
    mu_n = published['muw'].ravel()[3]
    sza = float(np.rad2deg(np.arcsin(np.sin(np.arccos(mu_n)) * 1.34)))
    bw = np.array([0.003])
    bp = bw / eta_n - bw
    res = ls2_main.ls2_invert(np.array([[0.004]]), np.array([[0.1]]),
                              np.array([0.02]), bw, bp[None], np.array([sza]),
                              np.array([500.]), lut, raman=False)
    a, bb = m.invert(0.004, 0.1, eta_n, mu_n)
    assert res.a[0, 0] == pytest.approx(a, rel=1e-6)
    assert res.bb[0, 0] == pytest.approx(bb, rel=1e-6)


def test_save_load_round_trip(tmp_path):
    lut_path = os.path.join(resources.files('ocpy'), 'data', 'LS2', 'LS2_LUT.npz')
    m = _truth_model()
    m.meta = {'note': 'x'}
    lut = m.to_lut(dict(np.load(lut_path)))
    refit.save(m, lut, tmp_path / 't.npz')
    lut2, m2 = refit.load(tmp_path / 't.npz')
    np.testing.assert_array_equal(lut2['a'], lut['a'])
    assert m2.muw_basis == ('inv', 'lin') and m2.meta == {'note': 'x'}
    np.testing.assert_allclose(m2.beta_b, m.beta_b)


def test_the_shipped_l23_table_runs():
    path = os.path.join(resources.files('ocpy'), 'data', 'LS2', 'LS2_LUT_L23_v1.npz')
    lut, model = refit.load(path)
    assert model.meta['corpus'].startswith('L23 X=1')
    assert model.muw_basis == ('inv', 'lin') and model.eta_degree == 2
    res = ls2_main.ls2_invert(np.array([[0.004, 0.002]]), np.array([[0.05, 0.08]]),
                              np.array([0.006, 0.06]), np.array([0.004, 0.002]),
                              np.array([[0.1, 0.08]]), np.array([30.]),
                              np.array([440., 555.]), lut, raman=False)
    assert np.all(np.isfinite(res.a)) and np.all(res.a > 0) and np.all(res.bb > 0)


# --- kappa (task 13) -----------------------------------------------------------

def test_fit_kappa_recovers_a_cubic_and_its_range():
    rng = np.random.default_rng(3)
    wave = np.array([440., 500., 670.])
    ratio = rng.uniform(0.01, 0.3, size=(500, 3))
    p = np.array([[2.0, -1.0, 0.2, 0.9], [1.0, 0.5, -0.3, 0.85], [0.0, 0.0, 0.1, 0.8]])
    kappa = np.stack([np.polyval(p[j], ratio[:, j]) for j in range(3)], axis=1)
    tab = refit.fit_kappa(wave, ratio, kappa, range_tol=0.0)
    assert tab.shape == (3, 7)
    np.testing.assert_array_equal(tab[:, 0], wave)
    np.testing.assert_allclose(tab[:, 1:5], p, atol=1e-8)
    np.testing.assert_allclose(tab[:, 5], ratio.min(axis=0))
    np.testing.assert_allclose(tab[:, 6], ratio.max(axis=0))
    wide = refit.fit_kappa(wave, ratio, kappa)          # default 1% widening
    assert np.all(wide[:, 5] < tab[:, 5]) and np.all(wide[:, 6] > tab[:, 6])


def test_kappa_eval_is_what_ls2_invert_applies():
    lut_path = os.path.join(resources.files('ocpy'), 'data', 'LS2', 'LS2_LUT.npz')
    published = dict(np.load(lut_path))
    k, use = refit.kappa_eval(published['kappa'], np.array([440., 502., 720.]),
                              np.array([0.1, 0.1, 0.01]))
    assert use[0] and not use[1] and not use[2]      # 502 row's range; >702 nm
    new = refit.with_kappa(published, published['kappa'][:50])
    assert new['kappa'].shape == (50, 7) and published['kappa'].shape == (101, 7)


def test_the_shipped_abk_table_covers_350_750_with_l23_ranges():
    path = os.path.join(resources.files('ocpy'), 'data', 'LS2', 'LS2_LUT_L23_abk_v1.npz')
    lut, model = refit.load(path)
    kap = lut['kappa']
    assert kap[0, 0] == 350. and kap[-1, 0] == 750. and kap.shape == (81, 7)
    assert 'refit from L23 X=1/X=2' in model.meta['kappa']
    # the published 502 nm hole is gone: kappa evaluates at a typical bb/a there
    k, use = refit.kappa_eval(kap, np.array([500., 700., 740.]),
                              np.array([0.08, 0.002, 0.0005]))
    assert use.all() and np.all((k > 0.7) & (k < 1.05))
