""" Module for spectral analysis of Tara Oceans data. """

import numpy as np
import pandas

from IPython import embed

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

    # Wavelengths
    wv_nm, keys = parse_wavelengths(tbl, flavor=flavor)

    # Read
    values, err_vals = [], []
    for key in keys:
        for ilist, ikey in zip([values,err_vals], 
                               [key, 'sig_'+key]):
            # Slurp
            val = tbl[ikey].values
            # Masked?
            val[np.isclose(val, -9999.)] = np.nan
            ilist.append(val)

    # Concatenate
    values = np.reshape(np.concatenate(values), 
                    (len(wv_nm), len(tbl)))
    err_vals = np.reshape(np.concatenate(err_vals), 
                    (len(wv_nm), len(tbl)))

    # Return
    return wv_nm, values, err_vals

def average_spectrum(tbl:pandas.DataFrame, flavor:str='ap'):
    wv_nm, values, err_vals = spectra_from_table(tbl, flavor=flavor)

    # Average
    avg_vals = np.nanmean(values, axis=1)
    avg_error = np.nanmean(err_vals, axis=1)

    # Cut
    gd_spec = np.isfinite(avg_vals)
    wv_nm = wv_nm[gd_spec]
    avg_vals = avg_vals[gd_spec]
    avg_error = avg_error[gd_spec]

    # Return
    return wv_nm, avg_vals, avg_error

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
        for ilist, ikey in zip([values,err_vals], 
                               [key, 'sig_'+key]):
            val = row[ikey]
            # Mask?
            val = np.nan if np.isclose(val, -9999.) else val
            # Save
            ilist.append(val)

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