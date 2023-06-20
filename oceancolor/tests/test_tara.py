""" Test methods in bayesian.py """

import os
from pkg_resources import resource_filename

import numpy as np
import pandas

from oceancolor.tara import io 
from oceancolor.tara import spectra

import pytest

db_present = os.path.isfile(io.db_name)

remote_data = pytest.mark.skipif(not db_present,
                                 reason='tests require TARA database ')

@remote_data
def test_load():
    # Inverse
    tara_db = io.load_tara_db()
    assert isinstance(tara_db, pandas.DataFrame)

# Load one spectrum
@remote_data
def test_load_spectrum():
    tara_db = io.load_tara_db()
    wv_nm, values, error = spectra.spectrum_from_row(tara_db.iloc[0])

    #
    assert isinstance(wv_nm, np.ndarray)
    assert np.sum(np.isfinite(values)) == len(values)

# Load spectra
@remote_data
def test_load_spectra():
    tara_db = io.load_tara_db()
    wv_nm, values, error = spectra.spectra_from_table(tara_db.loc[0:20])
    # Test
    assert isinstance(wv_nm, np.ndarray)
    assert values.shape[0] == len(wv_nm)

# Average spectrum
@remote_data
def test_average_spectrum():
    tara_db = io.load_tara_db()
    rio = tara_db[tara_db.cruise == 'Rio-BA']
    wv_nm, avg_spec, avg_err = spectra.average_spectrum(rio)
    # Test
    assert isinstance(wv_nm, np.ndarray)
    assert avg_spec.size == len(wv_nm)
    assert np.sum(np.isnan(avg_spec)) == 0

@remote_data
def test_single_value():
    tara_db = io.load_tara_db()
    value, sig = spectra.single_value(tara_db, 675.)

    # Test
    assert np.isclose(value[0], 0.01625)
    assert np.isclose(sig[0], 0.00465)