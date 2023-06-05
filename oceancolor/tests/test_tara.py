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
#tara_db = io.load_tara_db()