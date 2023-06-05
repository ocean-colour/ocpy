""" Module for spectral analysis of Tara Oceans data. """

import numpy as np
import pandas


def parse_wavelengths(inp, flavor:str='ap'):
    """ Parse wavelengths from a row/table of the Tara Oceans database. """
    keys, wv_nm = [], []

    for key in inp.keys():
        if key[0:2] == flavor:
            keys.append(key)
            # Wavelength
            wv_nm.append(float(key[2:]))
    # Recast
    wv_nm = np.array(wv_nm)
    keys = np.array(keys)

    # Sort
    srt = np.argsort(wv_nm)
    wv_nm = wv_nm[srt]
    keys = keys[srt]

    # Return
    return wv_nm, keys

def spectra_from_table(tbl:pandas.DataFrame, flavor:str='ap'):
    wv_nm, keys = parse_wavelengths(tbl, flavor=flavor)

def spectrum_from_row(row:pandas.Series, flavor:str='ap'):
    """ Load a spectrum from a row in the Tara Oceans database.

    Args:
        row (pandas.Series): 
            One row from the Tara Oceans database.
        flavor (str, optional): 
            Flavor of spectrum to load [ap, cp]  
            Defaults to 'ap'.

    Returns:
        tuple: wavelength (nm), values, error
    """
    # Wavelengths
    wv_nm, keys = parse_wavelengths(row, flavor=flavor)

    # Read
    values, err_vals = [], []
    for key in keys:
        # Slurp
        values.append(row[key])
        # Error
        err_key = 'sig_'+key
        err_vals.append(row[err_key])

    # Recast
    values = np.array(values)
    err_vals = np.array(err_vals)

    # Cut down
    gd_spec = np.isfinite(values)
    wv_nm = wv_nm[gd_spec]
    values = values[gd_spec]
    err_vals = err_vals[gd_spec]

    
    # Return
    return wv_nm, values, err_vals