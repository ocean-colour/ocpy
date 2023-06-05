""" Methods for I/O on Tara Oceans data. """
import os
import numpy as np

from pkg_resources import resource_filename
import pandas

from IPython import embed

db_name = os.path.join(resource_filename(
        'oceancolor', 'data'), 'Tara', 'Tara_APCP.parquet')

def load_tara_db():
    """ Load the Tara Oceans database. """
    # Get the file
    # Read
    df = pandas.read_parquet(db_name)
    # Return
    return df

def load_spectrum(row:pandas.Series, flavor:str='ap'):
    """ Load a spectrum """
    keys, wv_nm, values = [], [], []
    ii = 0


    for key in row.keys():
        if key[0:2] == flavor:
            keys.append(key)
            # Wavelength
            wv_nm.append(float(key[2:]))
            # Slurp
            values.append(row[key])

    # Recast
    wv_nm = np.array(wv_nm)
    values = np.array(values)
    keys = np.array(keys)

    # Cut down
    gd_spec = np.isfinite(values)
    wv_nm = wv_nm[gd_spec]
    values = values[gd_spec]
    keys = keys[gd_spec]

    # Sort
    srt = np.argsort(wv_nm)
    wv_nm = wv_nm[srt]
    values = values[srt]
    keys = keys[srt]

    # Error array
    err_vals = np.zeros_like(values)
    for kk, key in enumerate(keys):
        err_key = 'sig_'+key
        err_vals[kk] = row[err_key]
    
    # Return
    return wv_nm, values, err_vals
