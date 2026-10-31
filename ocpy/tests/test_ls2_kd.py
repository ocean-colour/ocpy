"""Tests for the Kd neural networks used by LS2.

The oracles are the authors' own reference vectors from
``Kd_NN_Distribution``, converted once to CSV by
``runs/LS2/convert_kd_nn_luts.py``:

* ``Kd_NN_test_run_MODIS_v1.1.csv`` -- 100 spectra (36 clear, 64 turbid),
  the release ocpy has always shipped;
* ``Kd_NN_test_run_MODIS_v1.3.csv`` -- the same 100 inputs through the
  authors' current MODIS network;
* ``Kd_NN_test_run_PACE_v2.3.csv`` -- one clear-water spectrum.

The PACE turbid branch has no reference output, so it is checked against a
literal, loop-for-loop transcription of ``Kd_NN_PACE.m`` instead; that pins the
vectorization and the input wiring, not the weights themselves.
"""

import pathlib

import numpy as np
import pandas
import pytest

from ocpy.ls2 import io as ls2_io
from ocpy.ls2 import kd_nn

FILES = pathlib.Path(__file__).parent / 'files'


def _reference(tag):
    """``(Rrs, sza, wave, Kd)`` from one of the authors' reference vectors."""
    df = pandas.read_csv(FILES / f'Kd_NN_test_run_{tag}.csv')
    rrs = df[[c for c in df.columns if c.startswith('Input Rrs')]].to_numpy()
    return (rrs, df['Input sza [deg]'].to_numpy(),
            df['Output Wavelength [nm]'].to_numpy(),
            df['Output Kd [1/m]'].to_numpy())


@pytest.mark.parametrize('tag', ['MODIS_v1.1', 'MODIS_v1.3', 'PACE_v2.3'])
def test_reference_vectors(tag):
    """Every release reproduces its own reference vector to 1e-11."""
    rrs, sza, wave, ref = _reference(tag)
    # Each reference row has its own output wavelength: take the diagonal.
    Kd = kd_nn.kd_nn(rrs, sza, wave, tag)
    idx = np.arange(len(ref))
    np.testing.assert_allclose(Kd[idx, idx], ref, rtol=1e-11)


def test_hard_coded_scalars():
    """The two values ocpy's tests have always pinned, now to 1e-9.

    They are rows 0 and 1 of the authors' v1.1 reference vector, one clear
    and one turbid, and they originally validated ``load_weights``' reshape
    order.  The default moved to v1.3 on 2026-10-04, so v1.1 is named here.
    """
    Rrs = [[0.00663061627778120, 0.00569886961307466, 0.00261783091270704,
            0.00206376550494829, 0.000155664525215032],
           [0.00278698717627185, 0.00555055435696759, 0.00965557516418457,
            0.0114125090066875, 0.00402413455723548]]

    Kd0 = kd_nn.Kd_NN_MODIS(Rrs[0], 30, 430, version='1.1')     # clear
    Kd1 = kd_nn.Kd_NN_MODIS(Rrs[1], 60, 531, version='1.1')     # turbid
    assert Kd0.shape == Kd1.shape == (1, 1)
    np.testing.assert_allclose(Kd0, 0.04600481236, rtol=1e-9)
    np.testing.assert_allclose(Kd1, 0.5623341777, rtol=1e-9)

    # The authors' current release gives 11% and 23% less on the same inputs.
    np.testing.assert_allclose(kd_nn.Kd_NN_MODIS(Rrs[0], 30, 430, version='1.3'),
                               0.0408230, rtol=1e-5)
    np.testing.assert_allclose(kd_nn.Kd_NN_MODIS(Rrs[1], 60, 531, version='1.3'),
                               0.4319691, rtol=1e-5)


@pytest.mark.parametrize('tag', ['MODIS_v1.1', 'MODIS_v1.3'])
def test_both_branches_are_exercised(tag):
    """The MODIS reference covers both branches; the switch is 488/547."""
    rrs, sza, wave, _ = _reference(tag)
    _, info = kd_nn.kd_nn(rrs, sza, wave, tag, return_branch=True)
    assert info['clear'].sum() == 36 and info['turbid'].sum() == 64
    np.testing.assert_array_equal(info['clear'], rrs[:, 1] / rrs[:, 3] >= 0.85)


def test_batch_equals_loop_of_scalars():
    """``(N, nb) x (L,)`` in one call equals the scalar wrapper cell by cell."""
    rrs, sza, _, _ = _reference('MODIS_v1.1')
    rrs, sza = rrs[::7], sza[::7]
    wave = np.array([412., 443., 490., 555., 670.])

    for version in ('1.1', '1.3'):
        block = kd_nn.kd_nn(rrs, sza, wave, f'MODIS_v{version}')
        loop = np.array([[kd_nn.Kd_NN_MODIS(r, s, w, version=version)[0, 0]
                          for w in wave] for r, s in zip(rrs, sza)])
        assert block.shape == (len(rrs), wave.size)
        np.testing.assert_allclose(block, loop, rtol=1e-14)


def _pace_literal(Rrs, sza, lam):
    """Loop-for-loop transcription of ``Kd_NN_PACE.m`` (v2.3), one spectrum."""
    with np.load(FILES.parent.parent / 'data' / 'LS2'
                 / 'Kd_NN_LUT_PACE_v2.3.npz') as npz:
        lut = {k: np.asarray(npz[k]) for k in npz.files}
    inputs = np.concatenate([Rrs, [sza, lam]])
    mu_s, std_s = lut['train_mean'], lut['train_std']
    if inputs[2] / inputs[5] >= 0.85:
        pre, ne, nc1, nc2 = 'clear', 12, 19, 17
        sel = list(range(10)) + [12, 13, 14]           # [1:10,13:15]
        x = inputs[list(range(10)) + [12, 13]]          # [1:10,13:14]
    else:
        pre, ne, nc1, nc2 = 'turbid', 14, 17, 9
        sel = list(range(15))                           # 1:15
        x = inputs
    mu, std = mu_s[sel], std_s[sel]
    w1 = lut[f'{pre}_w1'].reshape((nc1, ne), order='F')
    w2 = lut[f'{pre}_w2'].reshape((nc2, nc1), order='F')
    xn = np.array([(2 / 3) * ((x[j] - mu[j]) / std[j]) for j in range(ne)])

    def tansig(n):
        return 2 / (1 + np.exp(-2 * n)) - 1

    a = tansig(w1 @ xn + lut[f'{pre}_b1'])
    b = tansig(w2 @ a + lut[f'{pre}_b2'])
    y = lut[f'{pre}_wout'] @ b + lut[f'{pre}_bout'][0]
    return 10 ** (1.5 * y * std[-1] + mu[-1])


def test_pace_turbid_branch_against_literal_transcription():
    """The PACE turbid branch, which no reference vector reaches."""
    rng = np.random.default_rng(3)
    bands = np.array(kd_nn.load_network('PACE_v2.3').bands)
    # Green-peaked spectra: Rrs(490)/Rrs(560) well below 0.85.
    peak = np.exp(-0.5 * ((bands - 560.) / 60.) ** 2)
    rrs = (0.004 + 0.006 * rng.random((12, 1))) * (0.15 + peak)
    sza = rng.uniform(0., 70., 12)
    wave = np.array([440., 555., 650.])

    Kd, info = kd_nn.kd_nn(rrs, sza, wave, 'PACE_v2.3', return_branch=True)
    assert info['turbid'].all()
    literal = np.array([[_pace_literal(r, s, w) for w in wave]
                        for r, s in zip(rrs, sza)])
    np.testing.assert_allclose(Kd, literal, rtol=1e-12)
    assert np.all((Kd > 0.05) & (Kd < 20.))     # turbid, but not absurd


def test_negative_rrs_shape_and_flags(recwarn):
    """Negative Rrs gives a ``(1, 1)`` NaN with a warning, or a quiet NaN in batch.

    The old scalar path returned a bare ``np.nan`` here against a ``(1, 1)``
    array otherwise.
    """
    rrs, sza, wave, _ = _reference('MODIS_v1.1')
    clear_row = rrs[0].copy()
    bad = clear_row.copy()
    bad[0] = -1e-4

    with pytest.warns(UserWarning, match='Negative Rrs'):
        out = kd_nn.Kd_NN_MODIS(bad, 30., 443.)
    assert out.shape == (1, 1) and np.isnan(out[0, 0])

    # The clear branch never sees Rrs(667), so a negative value there is
    # ignored, exactly as in the authors' code.
    red_bad = clear_row.copy()
    red_bad[4] = -1e-4
    Kd, info = kd_nn.kd_nn(np.vstack([clear_row, bad, red_bad]), 30., 443.,
                           return_branch=True)
    assert np.isfinite(Kd[0, 0]) and np.isnan(Kd[1, 0])
    np.testing.assert_allclose(Kd[2, 0], Kd[0, 0], rtol=1e-14)
    np.testing.assert_array_equal(info['negative'], [False, True, False])


def test_non_finite_ratio_is_nan():
    """A spectrum whose switch ratio is NaN belongs to neither branch."""
    rrs, *_ = _reference('MODIS_v1.1')
    row = rrs[0].copy()
    row[3] = 0.
    nan_row = rrs[0].copy()
    nan_row[1] = np.nan
    Kd, info = kd_nn.kd_nn(np.vstack([row, nan_row]), 30., 443.,
                           return_branch=True)
    # Rrs(547) = 0 gives an infinite ratio, which is "clear".
    assert info['clear'][0]
    assert not info['clear'][1] and not info['turbid'][1]
    assert np.isnan(Kd[1, 0])


def test_weights_are_loaded_once(monkeypatch):
    """The scalar path no longer re-reads the weight files on every call."""
    kd_nn.load_network.cache_clear()
    first = kd_nn.load_network('MODIS_v1.1')
    assert kd_nn.load_network('MODIS_v1.1') is first

    def _boom():
        raise AssertionError('load_Kd_tables called on the hot path')

    monkeypatch.setattr(ls2_io, 'load_Kd_tables', _boom)
    rrs, sza, wave, ref = _reference('MODIS_v1.1')
    for i in range(5):
        np.testing.assert_allclose(kd_nn.Kd_NN_MODIS(rrs[i], sza[i], wave[i],
                                                     version='1.1'),
                                   [[ref[i]]], rtol=1e-11)


def test_legacy_load_weights_is_the_transpose():
    """The legacy ``load_weights``/``MLP_Kd`` path is the same v1.1 network.

    Settles the old "these could be backwards" comment: the legacy ``(ne,
    nc1)`` reshape is the transpose of MATLAB's column-major ``(nc1, ne)``.
    """
    net = kd_nn.load_network('MODIS_v1.1')
    for wtype, layers in (('clear', net.clear), ('turbid', net.turbid)):
        w1, b1, w2, b2, wout, bout = kd_nn.load_weights(wtype)
        np.testing.assert_array_equal(w1, layers.w1.T)
        np.testing.assert_array_equal(w2, layers.w2.T)
        np.testing.assert_array_equal(wout.ravel(), layers.wout)
        np.testing.assert_array_equal(b1, layers.b1)
        np.testing.assert_array_equal(b2, layers.b2)
        assert bout[0] == layers.bout


def test_bad_arguments():
    """Unknown networks and wrong band counts raise."""
    with pytest.raises(ValueError, match='Unknown Kd network'):
        kd_nn.kd_nn(np.ones(5) * 1e-3, 30., 443., 'SeaWiFS')
    with pytest.raises(ValueError, match='12 bands'):
        kd_nn.kd_nn(np.ones(5) * 1e-3, 30., 443., 'PACE_v2.3')
    with pytest.raises(ValueError, match='sza'):
        kd_nn.kd_nn(np.ones((3, 5)) * 1e-3, [30., 40.], 443.)


def test_the_default_is_v1_3():
    """Both entry points default to the authors' current MODIS release."""
    rrs, sza, wave, ref = _reference('MODIS_v1.3')
    np.testing.assert_allclose(kd_nn.Kd_NN_MODIS(rrs[0], sza[0], wave[0]),
                               [[ref[0]]], rtol=1e-11)
    np.testing.assert_allclose(kd_nn.kd_nn(rrs[:3], sza[:3], wave[0])[:, 0],
                               kd_nn.kd_nn(rrs[:3], sza[:3], wave[0],
                                           'MODIS_v1.3')[:, 0], rtol=0)
