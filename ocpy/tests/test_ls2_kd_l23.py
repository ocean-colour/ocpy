"""Tests for the L23-trained Kd networks (``ocpy.ls2.kd_l23``).

A small synthetic network pins the evaluation contract (round trip, feature
check, geometry handling, flags, interpolation); the shipped weights are then
checked to load and to give sane, positive Kd for a clear-water spectrum.
"""

import numpy as np
import pytest

from ocpy.ls2 import kd_l23


def _toy(geometry_input=False, bands=(443., 490., 510., 555., 670.)):
    """A 2-layer network whose output is easy to predict."""
    rng = np.random.default_rng(0)
    nb = len(bands) + (1 if geometry_input else 0)
    out = np.arange(400., 751., 5.)
    w1, b1 = rng.normal(size=(nb, 4)) * 0.1, np.zeros(4)
    w2, b2 = rng.normal(size=(4, out.size)) * 0.1, np.zeros(out.size)
    return kd_l23.KdL23Network(
        name='toy', layers=((w1, b1), (w2, b2)),
        x_mean=np.full(nb, 0.003), x_std=np.full(nb, 0.002),
        y_mean=np.log(np.linspace(0.02, 2.0, out.size)), y_std=np.full(out.size, 0.5),
        bands=np.asarray(bands), out_wave=out, geometry_input=geometry_input,
        domain=np.stack([np.zeros(nb), np.full(nb, 0.02)]),
        theta_s_trained=(0., 60.), theta_s_sanctioned=(0., 70.),
        meta={'note': 'test'})


def test_round_trip(tmp_path):
    net = _toy(geometry_input=True)
    p = tmp_path / 'toy.npz'
    kd_l23.save_network(net, p)
    back = kd_l23.from_npz(p)
    assert back.features == net.features and back.meta == {'note': 'test'}
    rrs = np.full((3, 5), 0.004)
    np.testing.assert_allclose(kd_l23.kd_l23(rrs, [0., 30., 60.], network=back),
                               kd_l23.kd_l23(rrs, [0., 30., 60.], network=net))


def test_a_feature_mismatch_is_refused(tmp_path):
    p = tmp_path / 'toy.npz'
    kd_l23.save_network(_toy(), p)
    data = dict(np.load(p))
    data['features'] = np.asarray(['Rrs_1'] * 5)
    np.savez(p, **data)
    with pytest.raises(ValueError, match='retrain'):
        kd_l23.from_npz(p)


def test_the_muw_factor_is_analytic_without_geometry_input():
    """With no geometry input, Kd * mu_w does not depend on the sun."""
    net = _toy()
    rrs = np.full((2, 5), 0.004)
    kd = kd_l23.kd_l23(rrs, [0., 60.], network=net)
    np.testing.assert_allclose(kd[0] * kd_l23.muw(0.), kd[1] * kd_l23.muw(60.))


def test_zenith_flags_and_nan_beyond_70():
    net = _toy()
    rrs = np.full((4, 5), 0.004)
    kd, f = kd_l23.kd_l23(rrs, [30., 65., 75., np.nan], network=net,
                          return_flags=True)
    assert list(f['extrapolated_sza']) == [False, True, False, False]
    assert list(f['unsupported_sza']) == [False, False, True, True]
    assert np.isfinite(kd[:2]).all() and np.isnan(kd[2:]).all()


def test_out_of_domain_flag():
    net = _toy()
    rrs = np.array([[0.004] * 5, [0.004, 0.004, 0.004, 0.004, 0.5]])
    _, f = kd_l23.kd_l23(rrs, 30., network=net, return_flags=True)
    assert list(f['out_of_domain']) == [False, True]


def test_wavelength_interpolation_is_in_ln_kd():
    net = _toy()
    rrs = np.full((1, 5), 0.004)
    full = kd_l23.kd_l23(rrs, 30., network=net)[0]
    mid = kd_l23.kd_l23(rrs, 30., [402.5, 800.], network=net)[0]
    assert mid[0] == pytest.approx(np.sqrt(full[0] * full[1]))
    assert np.isnan(mid[1])


def test_bad_arguments():
    net = _toy()
    with pytest.raises(ValueError, match='expects Rrs at 5 bands'):
        kd_l23.kd_l23(np.ones(4) * 1e-3, 30., network=net)
    with pytest.raises(ValueError, match='sza'):
        kd_l23.kd_l23(np.ones((3, 5)) * 1e-3, [30., 40.], network=net)
    with pytest.raises(ValueError, match='Unknown L23 Kd network'):
        kd_l23.load_network('L23_nope')


@pytest.mark.parametrize('name', sorted(kd_l23.NETWORKS))
def test_the_shipped_networks_load_and_give_clear_water_kd(name):
    net = kd_l23.load_network(name)
    assert net.meta['L23_X'] == 4 and net.theta_s_sanctioned == (0.0, 70.0)
    # A clear-water spectrum: Rrs falling from blue to red, as in L23.
    wave = np.arange(400., 751., 5.)
    rrs_full = 0.012 * np.exp(-((wave - 400.) / 120.) ** 2) + 1e-5
    rrs = np.interp(net.bands, wave, rrs_full)
    kd = kd_l23.kd_l23(rrs, 30., network=net)[0]
    assert np.all(np.isfinite(kd)) and np.all(kd > 0)
    # attenuation rises from the blue to the red, as pure water dictates
    assert kd[net.out_wave == 700.][0] > 10 * kd[net.out_wave == 440.][0]
