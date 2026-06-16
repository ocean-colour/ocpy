""" Tests for the ocpy.spectra package.

These tests are data-independent: they use small synthetic arrays so they
run everywhere. Class-level and data-backed adapter tests are added in later
phases of the refactor.
"""
import os

import numpy as np
import pytest

from ocpy.spectra import utils
from ocpy.spectra import Spectrum, SpectrumStack
from ocpy.spectra import io as spectra_io


def _panagea_available():
    """ True if the PANAGEA V3 directory can be resolved. """
    try:
        from ocpy.insitu import panagea
        panagea.panagea_path()
        return True
    except Exception:
        return False


def _loisel23_available():
    """ True if a Loisel+2023 Hydrolight file is present. """
    oc = os.getenv('OS_COLOR')
    if oc is None:
        return False
    return os.path.isfile(os.path.join(oc, 'Loisel2023', 'Hydrolight100.nc'))


def _tara_path():
    """ Path to the Tara APCP parquet, or None. """
    oc = os.getenv('OS_COLOR')
    if oc is None:
        return None
    path = os.path.join(oc, 'Tara', 'Tara_APCP.parquet')
    return path if os.path.isfile(path) else None


needs_panagea = pytest.mark.skipif(
    not _panagea_available(), reason='requires the PANAGEA V3 data')
needs_loisel23 = pytest.mark.skipif(
    not _loisel23_available(), reason='requires the Loisel2023 data')
needs_tara = pytest.mark.skipif(
    _tara_path() is None, reason='requires the Tara APCP data')


def _simple_spectrum(**kw):
    """ Build a small Spectrum for tests. """
    wv = np.array([400., 500., 600.])
    val = np.array([1., 2., 3.])
    err = np.array([0.1, 0.2, 0.3])
    base = dict(wavelength=wv, values=val, errors=err, units='1/sr',
                source='unit-test', lat=36.9, lon=-122.0, depth=5.0,
                quality={'qf_time': 0}, metadata={'station': 'M1'})
    base.update(kw)
    return Spectrum(**base)


# --------------------------------------------------------------------
# rebin (interpolation)
# --------------------------------------------------------------------
def test_rebin_interpolates_midpoints():
    wv = np.array([400., 500., 600.])
    val = np.array([1., 2., 3.])
    new_v, new_e = utils.rebin(wv, val, np.array([450., 550.]))
    assert np.allclose(new_v, [1.5, 2.5])
    # No errors supplied -> zeros.
    assert np.allclose(new_e, [0., 0.])


def test_rebin_out_of_range_is_nan():
    wv = np.array([400., 500., 600.])
    val = np.array([1., 2., 3.])
    new_v, _ = utils.rebin(wv, val, np.array([300., 700.]))
    assert np.all(np.isnan(new_v))


def test_rebin_propagates_error_curve():
    wv = np.array([400., 500., 600.])
    val = np.array([1., 2., 3.])
    err = np.array([0.1, 0.2, 0.3])
    _, new_e = utils.rebin(wv, val, np.array([450.]), err_vals=err)
    assert np.allclose(new_e, [0.15])


# --------------------------------------------------------------------
# rebin_to_grid (bin averaging) -- shape regression
# --------------------------------------------------------------------
def test_rebin_to_grid_shape_is_spectrum_major():
    # Two spectra, four native wavelengths -> (nspec, nwave).
    wv = np.array([410., 420., 510., 520.])
    values = np.array([[1., 3., 10., 30.],
                       [2., 4., 20., 40.]])
    errs = np.zeros_like(values)
    grid = np.array([400., 500., 600.])  # two bins: [400,500), [500,600)

    rwave, rvals, rerr = utils.rebin_to_grid(wv, values, errs, grid)

    # Bin centres.
    assert np.allclose(rwave, [450., 550.])
    # Output is (nspec, nbin) = (2, 2).
    assert rvals.shape == (2, 2)
    assert rerr.shape == (2, 2)
    # Bin means: spectrum 0 -> [2, 20]; spectrum 1 -> [3, 30].
    assert np.allclose(rvals, [[2., 20.], [3., 30.]])


def test_rebin_to_grid_empty_bin_is_nan():
    wv = np.array([410., 420.])
    values = np.array([[1., 3.]])
    errs = np.zeros_like(values)
    # Second bin [500,600) has no samples.
    grid = np.array([400., 500., 600.])
    _, rvals, rerr = utils.rebin_to_grid(wv, values, errs, grid)
    assert np.isnan(rvals[0, 1])
    assert np.isnan(rerr[0, 1])


def test_rebin_to_grid_promotes_1d():
    wv = np.array([410., 420.])
    values = np.array([1., 3.])
    errs = np.zeros_like(values)
    grid = np.array([400., 500.])
    _, rvals, _ = utils.rebin_to_grid(wv, values, errs, grid)
    assert rvals.shape == (1, 1)
    assert np.allclose(rvals, [[2.]])


def test_rebin_to_grid_ignores_nan():
    wv = np.array([410., 420., 430.])
    values = np.array([[1., np.nan, 3.]])
    errs = np.zeros_like(values)
    grid = np.array([400., 500.])
    _, rvals, _ = utils.rebin_to_grid(wv, values, errs, grid)
    # Mean of the two finite samples (1 and 3) -> 2.
    assert np.allclose(rvals, [[2.]])


# --------------------------------------------------------------------
# value_at
# --------------------------------------------------------------------
def test_value_at_scalar_returns_float():
    wv = np.array([400., 500., 600.])
    val = np.array([1., 2., 3.])
    out = utils.value_at(wv, val, 450.)
    assert isinstance(out, float)
    assert np.isclose(out, 1.5)


def test_value_at_array():
    wv = np.array([400., 500., 600.])
    val = np.array([1., 2., 3.])
    out = utils.value_at(wv, val, np.array([450., 550.]))
    assert np.allclose(out, [1.5, 2.5])


# --------------------------------------------------------------------
# common_grid
# --------------------------------------------------------------------
def test_common_grid_true_for_identical():
    a = np.array([400., 500., 600.])
    assert utils.common_grid([a, a.copy()]) is True


def test_common_grid_false_for_different_length():
    a = np.array([400., 500., 600.])
    b = np.array([400., 500.])
    assert utils.common_grid([a, b]) is False


def test_common_grid_false_for_different_values():
    a = np.array([400., 500., 600.])
    b = np.array([400., 500., 601.])
    assert utils.common_grid([a, b]) is False


def test_common_grid_trivial():
    assert utils.common_grid([]) is True
    assert utils.common_grid([np.array([1., 2.])]) is True


# --------------------------------------------------------------------
# align_to_grid
# --------------------------------------------------------------------
def test_align_to_grid_stacks_ragged():
    # Two spectra on different native grids and lengths.
    wv1 = np.array([400., 500., 600.])
    v1 = np.array([1., 2., 3.])
    wv2 = np.array([400., 450., 500., 550., 600.])
    v2 = np.array([10., 15., 20., 25., 30.])
    grid = np.array([450., 550.])

    vals, errs = utils.align_to_grid([wv1, wv2], [v1, v2], grid)
    assert vals.shape == (2, 2)
    assert errs.shape == (2, 2)
    assert np.allclose(vals[0], [1.5, 2.5])
    assert np.allclose(vals[1], [15., 25.])


# --------------------------------------------------------------------
# Spectrum class
# --------------------------------------------------------------------
def test_spectrum_len_and_repr():
    s = _simple_spectrum()
    assert len(s) == 3
    assert 'Spectrum' in repr(s)
    assert '1/sr' in repr(s)


def test_spectrum_validates_length():
    with pytest.raises(ValueError):
        Spectrum(wavelength=np.array([1., 2.]), values=np.array([1.]))


def test_spectrum_sorts_wavelength():
    s = Spectrum(wavelength=np.array([600., 400., 500.]),
                 values=np.array([3., 1., 2.]))
    assert np.allclose(s.wavelength, [400., 500., 600.])
    assert np.allclose(s.values, [1., 2., 3.])


def test_spectrum_rebin_interp():
    s = _simple_spectrum()
    out = s.rebin(np.array([450., 550.]))
    assert isinstance(out, Spectrum)
    assert np.allclose(out.values, [1.5, 2.5])
    # Provenance carried over.
    assert out.source == 'unit-test'
    assert out.units == '1/sr'


def test_spectrum_rebin_bin():
    s = _simple_spectrum(errors=None)
    out = s.rebin(np.array([350., 450., 650.]), method='bin')
    assert out.errors is None
    # First bin [350,450) -> value 1; second [450,650) -> mean(2,3)=2.5.
    assert np.allclose(out.values, [1., 2.5])


def test_spectrum_value_at():
    s = _simple_spectrum()
    assert np.isclose(s.value_at(450.), 1.5)


def test_spectrum_xarray_roundtrip():
    s = _simple_spectrum()
    ds = s.to_xarray()
    # Wavelength must be a data variable, NOT a coordinate (design Q15).
    assert 'wavelength' in ds.data_vars
    assert 'wavelength' not in ds.coords
    back = Spectrum.from_xarray(ds)
    assert np.allclose(back.wavelength, s.wavelength)
    assert np.allclose(back.values, s.values)
    assert np.allclose(back.errors, s.errors)
    assert back.units == s.units
    assert back.source == s.source
    assert back.lat == s.lat
    assert back.quality == s.quality
    assert back.metadata == s.metadata


def test_spectrum_xarray_no_errors():
    s = _simple_spectrum(errors=None)
    back = Spectrum.from_xarray(s.to_xarray())
    assert back.errors is None


def test_spectrum_netcdf_roundtrip(tmp_path):
    s = _simple_spectrum()
    path = str(tmp_path / 'spec.nc')
    s.to_netcdf(path)
    back = Spectrum.read_netcdf(path)
    assert np.allclose(back.values, s.values)
    assert back.source == s.source
    assert back.metadata == s.metadata


def test_spectrum_plot_smoke():
    import matplotlib
    matplotlib.use('Agg')
    s = _simple_spectrum()
    ax = s.plot()
    assert ax is not None
    assert len(ax.lines) == 1


# --------------------------------------------------------------------
# SpectrumStack class
# --------------------------------------------------------------------
def test_stack_container_protocol():
    a = _simple_spectrum(source='a')
    b = _simple_spectrum(source='b')
    stack = SpectrumStack([a, b])
    assert len(stack) == 2
    assert stack[0].source == 'a'
    assert [s.source for s in stack] == ['a', 'b']


def test_stack_rejects_non_spectrum():
    with pytest.raises(TypeError):
        SpectrumStack([1, 2])


def test_stack_is_gridded_true():
    stack = SpectrumStack([_simple_spectrum(), _simple_spectrum()])
    assert stack.is_gridded is True
    wv, vals, errs = stack.as_array()
    assert vals.shape == (2, 3)


def test_stack_ragged_detection_and_rebin():
    a = _simple_spectrum()
    b = Spectrum(wavelength=np.array([400., 450., 500., 600.]),
                 values=np.array([1., 1.5, 2., 3.]))
    stack = SpectrumStack([a, b])
    assert stack.is_gridded is False
    with pytest.raises(ValueError):
        stack.as_array()
    gridded = stack.rebin(np.array([450., 550.]))
    assert gridded.is_gridded is True


def test_stack_provenance_arrays():
    a = _simple_spectrum(lat=10.0)
    b = _simple_spectrum(lat=20.0)
    stack = SpectrumStack([a, b])
    assert np.allclose(stack.lats, [10.0, 20.0])
    assert list(stack.sources) == ['unit-test', 'unit-test']


def test_stack_xarray_roundtrip_gridded():
    stack = SpectrumStack([_simple_spectrum(source='a'),
                           _simple_spectrum(source='b')])
    ds = stack.to_xarray()
    assert 'wavelength' in ds.data_vars
    assert 'wavelength' not in ds.coords
    back = SpectrumStack.from_xarray(ds)
    assert len(back) == 2
    assert back[0].source == 'a'
    assert np.allclose(back[1].values, stack[1].values)


def test_stack_xarray_roundtrip_ragged():
    a = _simple_spectrum()
    b = Spectrum(wavelength=np.array([400., 450., 500., 600.]),
                 values=np.array([1., 1.5, 2., 3.]), source='ragged')
    stack = SpectrumStack([a, b])
    back = SpectrumStack.from_xarray(stack.to_xarray())
    assert len(back) == 2
    # The NaN pad must be stripped: member b keeps its 4 channels.
    assert len(back[1]) == 4
    assert back[1].source == 'ragged'
    assert np.allclose(back[0].wavelength, a.wavelength)


def test_stack_netcdf_roundtrip(tmp_path):
    stack = SpectrumStack([_simple_spectrum(source='a'),
                           _simple_spectrum(source='b')])
    path = str(tmp_path / 'stack.nc')
    stack.to_netcdf(path)
    back = SpectrumStack.read_netcdf(path)
    assert len(back) == 2
    assert back[0].source == 'a'


# --------------------------------------------------------------------
# Source adapters (data-dependent; skip-guarded)
# --------------------------------------------------------------------
@needs_panagea
def test_from_panagea():
    from ocpy.insitu import panagea
    df = panagea.load('rrs')
    obs = df.index[0]
    s = spectra_io.from_panagea(df, obs, kind='rrs')
    assert isinstance(s, Spectrum)
    assert len(s) > 0
    assert s.units == '1/sr'
    assert s.source == 'PANAGEA'
    # Wavelengths sorted ascending.
    assert np.all(np.diff(s.wavelength) > 0)
    # xarray round trip works on a real spectrum.
    back = Spectrum.from_xarray(s.to_xarray())
    assert np.allclose(back.values, s.values)


@needs_panagea
def test_stack_from_panagea_subset():
    from ocpy.insitu import panagea
    df = panagea.load('rrs')
    ids = df.index[:5]
    stack = spectra_io.stack_from_panagea(df, kind='rrs', ids=ids)
    assert isinstance(stack, SpectrumStack)
    assert len(stack) <= 5


@needs_loisel23
def test_from_loisel23():
    from ocpy.hydrolight import loisel23
    ds = loisel23.load_ds(1, 0)
    s = spectra_io.from_loisel23(ds, 0, var='Rrs', units='1/sr')
    assert isinstance(s, Spectrum)
    assert len(s) == ds['Lambda'].size
    assert s.source == 'Loisel2023'
    assert s.units == '1/sr'
    assert s.errors is None


@needs_loisel23
def test_stack_from_loisel23_is_gridded():
    from ocpy.hydrolight import loisel23
    ds = loisel23.load_ds(1, 0)
    stack = spectra_io.stack_from_loisel23(ds, var='Rrs', indices=range(4))
    assert len(stack) == 4
    # All share the Lambda grid -> gridded.
    assert stack.is_gridded is True
    wv, vals, errs = stack.as_array()
    assert vals.shape == (4, ds['Lambda'].size)


@needs_tara
def test_from_tara():
    import pandas
    tbl = pandas.read_parquet(_tara_path())
    s = spectra_io.from_tara(tbl.iloc[0], flavor='ap')
    assert isinstance(s, Spectrum)
    assert len(s) > 0
    assert s.units == '1/m'
    assert s.source == 'Tara'
    # Provenance from real columns.
    assert s.lat is not None and s.lon is not None
    assert s.date is not None  # parsed from the 'datetime' column
    assert 'cruise' in s.metadata


@needs_tara
def test_stack_from_tara_subset():
    import pandas
    tbl = pandas.read_parquet(_tara_path())
    stack = spectra_io.stack_from_tara(tbl.iloc[:20], flavor='ap')
    assert isinstance(stack, SpectrumStack)
    assert len(stack) <= 20
    # The ap grid is shared across rows -> gridded.
    assert stack.is_gridded is True


# --------------------------------------------------------------------
# Edge cases / gap-fillers
# --------------------------------------------------------------------
def test_align_to_grid_with_errors():
    wv1 = np.array([400., 500., 600.])
    v1 = np.array([1., 2., 3.])
    e1 = np.array([0.1, 0.2, 0.3])
    vals, errs = utils.align_to_grid([wv1], [v1], np.array([450., 550.]),
                                     list_of_err=[e1])
    assert np.allclose(vals[0], [1.5, 2.5])
    assert np.allclose(errs[0], [0.15, 0.25])


def test_value_at_out_of_range_is_nan():
    wv = np.array([400., 500., 600.])
    val = np.array([1., 2., 3.])
    assert np.isnan(utils.value_at(wv, val, 700.))


def test_spectrum_rebin_interp_keeps_none_errors():
    s = _simple_spectrum(errors=None)
    out = s.rebin(np.array([450., 550.]))
    assert out.errors is None
    assert np.allclose(out.values, [1.5, 2.5])


def test_spectrum_metadata_not_shared():
    # dataclass default_factory must give each instance its own dict.
    a = Spectrum(wavelength=np.array([1., 2.]), values=np.array([1., 2.]))
    b = Spectrum(wavelength=np.array([1., 2.]), values=np.array([1., 2.]))
    a.metadata['x'] = 1
    assert b.metadata == {}


def test_empty_stack_as_array():
    stack = SpectrumStack([])
    assert len(stack) == 0
    assert stack.is_gridded is True
    wv, vals, errs = stack.as_array()
    assert wv.size == 0
    assert vals.shape == (0, 0)


def test_stack_netcdf_roundtrip_ragged(tmp_path):
    a = _simple_spectrum()
    b = Spectrum(wavelength=np.array([400., 450., 500., 600.]),
                 values=np.array([1., 1.5, 2., 3.]), source='ragged')
    stack = SpectrumStack([a, b])
    path = str(tmp_path / 'ragged.nc')
    stack.to_netcdf(path)
    back = SpectrumStack.read_netcdf(path)
    assert len(back) == 2
    assert len(back[1]) == 4  # NaN pad stripped on read
    assert back[1].source == 'ragged'
