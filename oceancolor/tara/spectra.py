""" Module for spectral analysis of Tara Oceans data. """

import numpy as np
import pandas
from scipy.interpolate import interp1d

from IPython import embed

def parse_wavelengths(inp, flavor:str='ap'):
    """ Parse wavelengths from a row/table of the Tara Oceans database. 

    Args:
        inp (pandas.Series or pandas.DataFrame):
            One row or table of the Tara Oceans database.
        flavor (str, optional):
            Flavor of spectrum to load [ap, cp]

    Returns:
        tuple: wavelengths (nm) [np.ndarray], keys [np.ndarray]
    """
    keys, wv_nm = [], []

    for key in inp.keys():
        if key[0:2] == flavor:
            # Skip a few 
            if key in ['ap_n', 'ap676_lh', 'cp_n']:
                continue
            # Ends in sd
            if key[-2:] == 'sd':
                continue
            # Keep
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

def spectbl_from_keys(tbl:pandas.DataFrame, keys:np.ndarray):
    """
    Extracts values and error values from a pandas DataFrame based on a given set of keys.

    Args:
        tbl (pandas.DataFrame): The input DataFrame containing the data.
        keys (np.ndarray): An array of keys specifying the columns to extract from the DataFrame.

    Returns:
        tuple: A tuple containing two numpy arrays - values and error values.
               The values array has shape (len(keys), len(tbl)) and contains the extracted values.
               The error values array has the same shape and contains the extracted error values.
    """

    # Read
    values, err_vals = [], []
    for key in keys:
        err_key = 'sig_'+key if 'sig_'+key in tbl.keys() else key+'_sd'
        # Hack for integer keys
        if err_key not in tbl.keys() and 'sd' in err_key:
            # Try adding in a .0
            err_key = err_key.replace('_sd', '.0_sd')
        # Do it
        for ilist, ikey in zip([values,err_vals], 
                               [key, err_key]):
            # Slurp
            val = tbl[ikey].values
            # Masked?
            val[np.isclose(val, -9999.)] = np.nan
            ilist.append(val)

    # Concatenate
    values = np.reshape(np.concatenate(values), 
                    (len(keys), len(tbl)))
    err_vals = np.reshape(np.concatenate(err_vals), 
                    (len(keys), len(tbl)))

    return values, err_vals

def spectra_from_table(tbl:pandas.DataFrame, flavor:str='ap'):
    """ Load spectra from a table of the Tara Oceans database.

    Args:
        tbl (pandas.DataFrame): 
            Table of the Tara Oceans database.
        flavor (str, optional): 
            Flavor of spectrum to load [ap, cp]

    Returns:
        tuple: wavelengths (nm), values, error
    """

    # Wavelengths
    wv_nm, keys = parse_wavelengths(tbl, flavor=flavor)

    # Do it
    values, err_vals = spectbl_from_keys(tbl, keys)

    # Return
    return wv_nm, values, err_vals

def average_spectrum(tbl:pandas.DataFrame, flavor:str='ap'):
    """ Average spectrum from a table of the Tara Oceans database.

    Note that NaN in the data are ignored.

    Args:
        tbl (pandas.DataFrame): 
            Table of the Tara Oceans database.
        flavor (str, optional): 
            Flavor of spectrum to load [ap, cp]

    Returns:
        tuple: wavelengths (nm), values, error
    """
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

def spectrum_from_row(row:pandas.Series, flavor:str='ap',
                      keep_nan:bool=False):
    """ Load a spectrum from a row in the Tara Oceans database.

    Args:
        row (pandas.Series): 
            One row from the Tara Oceans database.
        flavor (str, optional): 
            Flavor of spectrum to load [ap, cp]  
            Defaults to 'ap'.
        keep_nan (bool, optional):
            Keep NaN values in the spectrum.  Default is False

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
    if not keep_nan:
        gd_spec = np.isfinite(values)
        wv_nm = wv_nm[gd_spec]
        values = values[gd_spec]
        err_vals = err_vals[gd_spec]

    
    # Return
    return wv_nm, values, err_vals

def single_value(tbl:pandas.DataFrame, wv_cen:float, 
                 wv_delta:float=10., flavor:str='ap'):
    """ Return ap or cp at a single wavelength from a table of the Tara Oceans database

    This is generated by taking an average of 
     +/- wv_delta/2. around wv_cen

    Args:
        tbl (pandas.DataFrame): Table of the Tara Oceans database.
        wv_cen (float): Central wavelength (nm)
        wv_delta (float, optional): Range of wavelength to average over. Defaults to 10..
        flavor (str, optional): 
            Flavor of spectrum [ap, cp]  
            Defaults to 'ap'.

    Returns:
        tuple: value, error [np.ndarray, np.ndarray]
    """
    # Wavelengths
    wv_nm, keys = parse_wavelengths(tbl, flavor=flavor)

    # Cut
    gd_wv = np.where((wv_nm >= wv_cen-wv_delta/2.) & (
        wv_nm <= wv_cen+wv_delta/2.))[0]
    gd_keys = keys[gd_wv]

    # Build
    values, err_vals = spectbl_from_keys(tbl, gd_keys)

    # Average
    value = np.nanmean(values, axis=0)
    sig = np.nanmean(err_vals, axis=0)
    
    # Return
    return value, sig
