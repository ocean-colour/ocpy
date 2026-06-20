""" Tests for the PANGAEA dataset API (ocpy.insitu.pangaea).

The data files are large and not packaged, so the data-dependent tests are
skipped automatically when the PANGAEA V3 directory is unavailable. The
parsing/header tests run everywhere using synthetic inputs.
"""
import os

import numpy as np
import pandas
import pytest

from ocpy.insitu import pangaea


def _data_available():
    """ Return True if the PANGAEA V3 directory can be resolved. """
    try:
        pangaea.pangaea_path()
        return True
    except FileNotFoundError:
        return False


data_present = _data_available()
needs_data = pytest.mark.skipif(
    not data_present, reason='requires the PANGAEA V3 data directory')


# --------------------------------------------------------------------
# Data-independent tests (always run)
# --------------------------------------------------------------------
def test_file_catalog():
    cat = pangaea.file_catalog()
    assert len(cat) == 7
    assert cat['rrs'] == 'insitudb_rrs_V3.tab'
    # Should be a copy: mutating it must not affect the module dict.
    cat['rrs'] = 'x'
    assert pangaea.FILE_CATALOG['rrs'] == 'insitudb_rrs_V3.tab'


def test_find_header_end(tmp_path):
    # Build a tiny PANGAEA-style file with a 3-line header.
    p = tmp_path / 'mini.tab'
    p.write_text(
        '/* DATA DESCRIPTION:\n'
        'Citation:\tfoo\n'
        '*/\n'
        'ID (idx)\tDate/Time\tRRS [1/sr] (at 443nm)\n'
        '1\t1997-01-02T12:48\t0.01\n', encoding='utf-8')
    n = pangaea.find_header_end(str(p))
    assert n == 3
    df = pandas.read_csv(p, sep='\t', skiprows=n)
    assert list(df.columns)[0] == 'ID (idx)'


def test_parse_native_spectral():
    m = pangaea._parse_column('RRS [1/sr] (at 349.01nm)')
    assert m['role'] == 'spectral'
    assert m['variable'] == 'rrs'
    assert m['wavelength'] == pytest.approx(349.01)
    assert m['unit'] == '1/sr'
    assert m['friendly'] == 'rrs_349.01'


def test_parse_satband_pair():
    val = pangaea._parse_column('RRS [1/sr] (at +-2nm of OLCI-S3A band 11)')
    assert val['role'] == 'spectral_band'
    assert val['variable'] == 'rrs'
    assert val['sensor'] == 'OLCI-S3A'
    assert val['band'] == '11'
    lam = pangaea._parse_column('Lambda [nm] (assigned to OLCI-S3A band 11)')
    assert lam['role'] == 'lambda'
    assert (lam['sensor'], lam['band']) == ('OLCI-S3A', '11')
    # The value and lambda columns must produce the same band token,
    # which is what _pair_lambda_columns relies on.
    assert val['sensor'] == lam['sensor'] and val['band'] == lam['band']


def test_parse_iop_satband_lambda_with_prefix():
    lam = pangaea._parse_column(
        'Lambda [nm] (aph, assigned to MERIS band MER1)')
    assert lam['role'] == 'lambda'
    assert (lam['sensor'], lam['band']) == ('MERIS', 'MER1')


def test_parse_metadata_and_provenance():
    assert pangaea._parse_column('ID (idx)')['role'] == 'id'
    assert pangaea._parse_column('Date/Time')['friendly'] == 'date_time'
    assert pangaea._parse_column('Latitude')['friendly'] == 'lat'
    assert pangaea._parse_column('Depth water [m] (...)')['friendly'] \
        == 'depth_m'
    prov = pangaea._parse_column('Comment (chla_hplc_dataset)')
    assert prov['role'] == 'provenance'
    assert prov['friendly'] == 'chla_hplc_dataset'


def test_parse_chla_methods_kept_separate():
    hplc = pangaea._parse_column(
        'Chl a [mg/m**3] (High Performance Liquid Chrom...)')
    fluor = pangaea._parse_column(
        'Chl a [mg/m**3] (Chlorophyll a, fluorometric o...)')
    assert hplc['friendly'] == 'chla_hplc'
    assert fluor['friendly'] == 'chla_fluor'


def test_build_columns_unique():
    # Two distinct raw names that both reduce to the friendly 'dataset'
    # (this really happens in the IOP file's provenance comments).
    raw = ['Comment (dataset)',
           'Comment (dataset (tss = tsm in article))']
    rename, meta = pangaea._build_columns(raw)
    # Friendly names must be disambiguated to stay unique.
    assert len(set(rename.values())) == 2


def test_pair_lambda_columns():
    raw = ['RRS [1/sr] (at +-2nm of OLCI-S3A band 11)',
           'Lambda [nm] (assigned to OLCI-S3A band 11)']
    rename, meta = pangaea._build_columns(raw)
    meta = pangaea._pair_lambda_columns(meta)
    val_name = rename[raw[0]]
    lam_name = rename[raw[1]]
    assert meta[val_name]['lambda_col'] == lam_name


def test_spectrum_native_synthetic():
    # Build a synthetic loaded frame (native-wavelength style).
    df = pandas.DataFrame({'rrs_443': [0.01], 'rrs_490': [0.02]},
                          index=pandas.Index([5], name='ID'))
    df.attrs['columns'] = {
        'rrs_443': {'variable': 'rrs', 'role': 'spectral',
                    'wavelength': 443.0},
        'rrs_490': {'variable': 'rrs', 'role': 'spectral',
                    'wavelength': 490.0},
    }
    spec = pangaea.spectrum(df, 5, kind='rrs')
    assert list(spec.index) == [443.0, 490.0]
    assert spec.loc[490.0] == pytest.approx(0.02)


def test_spectrum_satband_synthetic():
    # Sat-band style: wavelength comes from the per-row Lambda column.
    df = pandas.DataFrame(
        {'rrs_S_band1': [0.05], 'lambda_S_band1': [444.2]},
        index=pandas.Index([9], name='ID'))
    df.attrs['columns'] = {
        'rrs_S_band1': {'variable': 'rrs', 'role': 'spectral_band',
                        'sensor': 'S', 'band': '1',
                        'lambda_col': 'lambda_S_band1'},
        'lambda_S_band1': {'variable': None, 'role': 'lambda',
                           'sensor': 'S', 'band': '1'},
    }
    spec = pangaea.spectrum(df, 9, kind='rrs')
    assert list(spec.index) == [444.2]
    assert spec.iloc[0] == pytest.approx(0.05)


# --------------------------------------------------------------------
# Data-dependent tests (skipped if the V3 directory is absent)
# --------------------------------------------------------------------
@needs_data
def test_load_chla():
    df = pangaea.load('chla')
    assert isinstance(df, pandas.DataFrame)
    assert df.index.name == 'ID'
    # Friendly names present.
    for col in ('lat', 'lon', 'date_time', 'chla_hplc', 'chla_fluor'):
        assert col in df.columns
    assert df.attrs['kind'] == 'chla'


@needs_data
def test_load_rrs_and_spectrum():
    df = pangaea.load('rrs')
    obs = df.index[0]
    spec = pangaea.spectrum(df, obs, kind='rrs')
    assert isinstance(spec, pandas.Series)
    assert spec.index.is_monotonic_increasing
    assert np.all(np.isfinite(spec.values))


@needs_data
def test_load_rrs_satband_spectrum():
    df = pangaea.load('rrs_sat2')
    assert df.attrs['is_satband'] is True
    obs = df.index[0]
    spec = pangaea.spectrum(df, obs, kind='rrs')
    # Wavelengths should fall in a sensible optical range.
    if len(spec) > 0:
        assert spec.index.min() > 300.0
        assert spec.index.max() < 1200.0


@needs_data
def test_extract_hyperspectral():
    from ocpy.spectra import Spectrum, SpectrumStack
    # A high threshold keeps the selection small and fast.
    data = pangaea.extract_hyperspectral(nband=150)
    assert isinstance(data, dict) and len(data) > 0
    # Every record must have a hyperspectral Rrs Spectrum and a stack.
    for obs_id, rec in data.items():
        assert isinstance(rec['rrs'], Spectrum)
        assert rec['n_rrs'] > 150
        assert len(rec['rrs']) == rec['n_rrs']
        assert rec['rrs'].units == '1/sr'
        assert isinstance(rec['iops'], SpectrumStack)
        assert rec['id'] == obs_id
    # The IOP members (when present) should be 1/m absorption/scatter.
    for rec in data.values():
        for s in rec['iops']:
            assert s.units == '1/m'
            assert s.metadata.get('kind') in ('aph', 'acdom', 'bbp', 'kd')
