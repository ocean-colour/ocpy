""" Utilities for spectra. """

import numpy as np
from scipy.interpolate import interp1d


def rebin(wv_nm:np.ndarray, values:np.ndarray, 
          err_vals:np.ndarray, wv_grid:np.ndarray):
    """ Rebin a spectrum to a new wavelength grid.

    Args:
        wv_nm (np.ndarray): Wavelengths (nm)
        values (np.ndarray): Values
        err_vals (np.ndarray): Error values
        wv_grid (np.ndarray): New wavelength grid

    Returns:
        tuple: values, error [np.ndarray, np.ndarray]
    """
    # Interpolate
    f_values = interp1d(wv_nm, values, 
        bounds_error=False, fill_value=np.nan)
    f_err = interp1d(wv_nm, err_vals, 
        bounds_error=False, fill_value=np.nan)

    # Evaluate
    new_values = f_values(wv_grid)
    new_err = f_err(wv_grid)

    # Return
    return new_values, new_err
