""" Utilities for spectra. """

import numpy as np
from scipy.interpolate import interp1d

from IPython import embed

def rebin(wv_nm:np.ndarray, values:np.ndarray, 
          wv_grid:np.ndarray,
          err_vals:np.ndarray=None): 
    """ Rebin a spectrum to a new wavelength grid.

    This uses interpolation.
    The method below implements without binning

    Args:
        wv_nm (np.ndarray): Wavelengths (nm)
        values (np.ndarray): Values
        wv_grid (np.ndarray): New wavelength grid
        err_vals (np.ndarray, optional): Error values

    Returns:
        tuple: values, error [np.ndarray, np.ndarray]
    """
    # Interpolate
    f_values = interp1d(wv_nm, values, 
        bounds_error=False, fill_value=np.nan)
    if err_vals is not None:
        f_err = interp1d(wv_nm, err_vals, 
            bounds_error=False, fill_value=np.nan)

    # Evaluate
    new_values = f_values(wv_grid)
    if err_vals is not None:
        new_err = f_err(wv_grid)
    else:
        new_err = np.zeros_like(new_values)

    # Return
    return new_values, new_err


def rebin_to_grid(wv_nm:np.ndarray, values:np.ndarray, 
                  err_vals:np.ndarray, wv_grid:np.ndarray):
    """ Rebin spectra to a new wavelength grid.

    Simple nearest neighbor binning (no interpolation)

    Args:
        wv_nm (np.ndarray): Wavelengths (nm)
        values (np.ndarray): Values (nwave, nspec)
        err_vals (np.ndarray): Error values (nwave, nspec)
        wv_grid (np.ndarray): New wavelength grid

    Returns:
        tuple: wave, values, error [np.ndarray (nwv), np.ndarray (nspec,nwv), np.ndarray]
    """
    gd_values = np.isfinite(values)
    mask = gd_values.astype(int)

    # Loop on wv_grid
    rebin_values = np.zeros((values.shape[1], wv_grid.size-1))
    rebin_err = np.zeros((values.shape[1], wv_grid.size-1))
    rebin_wave = np.zeros(wv_grid.size-1)
    
    for iwv in range(wv_grid.size-1):
        w0 = wv_grid[iwv]
        w1 = wv_grid[iwv+1]
        rebin_wave[iwv] = (w0+w1)/2.
        # In grid?
        gd = np.where((wv_nm >= w0) & (wv_nm < w1))[0]

        # Check
        if len(gd) == 0:
            rebin_err[:,iwv] = np.nan
            continue

        # Add em in
        mask_sum = np.sum(mask[gd],axis=0)
        isum = np.nansum(values[gd]*mask[gd], axis=0) / mask_sum
        esum = np.nansum(err_vals[gd]*mask[gd], axis=0) / mask_sum

        # Fill
        rebin_values[:,iwv] = isum
        rebin_err[:,iwv] = esum

    # Return
    return rebin_wave, rebin_values, rebin_err