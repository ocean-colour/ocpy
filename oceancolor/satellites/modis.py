""" Items related to MODIS """
import os
import numpy as np
from importlib.resources import files

from scipy.stats import sigmaclip

import pandas




# MODIS Aqua -- derived from https://seabass.gsfc.nasa.gov/search/?search_type=Perform%20Validation%20Search&val_sata=1&val_products=11&val_source=0
#  See MODIS_error.ipynb
#wv: 412, std=0.00141 sr^-1, rel_std=10.88%
#wv: 443, std=0.00113 sr^-1, rel_std=9.36%
#wv: 488, std=0.00113 sr^-1, rel_std=0.68%
#wv: 531, std=0.00102 sr^-1, rel_std=0.19%
#wv: 547, std=0.00117 sr^-1, rel_std=0.19%
#wv: 555, std=0.00120 sr^-1, rel_std=0.22%
#wv: 667, std=0.00056 sr^-1, rel_std=6.22%
#wv: 678, std=0.00060 sr^-1, rel_std=4.15%

modis_wave = np.array([412, 443, 469, 488, 531, 547, 555, 645, 667, 678, 748])# , 859, 869] # nm
modis_aqua_error = np.array([0.00141, 0.00113, 
                    0.00113,  # Assumed for 469
                    0.00113, 0.00102, 0.00117, 0.00120, 
                    0.00070,  # Assumed for 645
                    0.00056, 0.00060,
                    0.00060,  # Assumed for 748
                    ])

def load_matchups():
    """
    Load the MODIS matchups.

    Returns:
        pandas.DataFrame: DataFrame containing the MODIS matchups.
    """
    modis_file = files('oceancolor').joinpath(
        os.path.join('data', 'satellites', 'MODIS_matchups_rrs.csv'))
    modis = pandas.read_csv(modis_file, comment='#')
    return modis

def calc_stats(modis:pandas.DataFrame, wv:int, sig_cut:float=4.):
    """
    Calculate error statistics for the MODIS data using
    the difference between the Aqua and in-situ Rrs.
        Performs sigma clipping on the distribution.

    Parameters:
        modis (pandas.DataFrame): DataFrame containing MODIS data.
        wv (int): Wavelength of interest.
        sig_cut (float, optional): Sigma cut-off value. Default is 4.

    Returns:
    tuple: A tuple containing the following elements:
        - diff (pandas.Series): Difference between 'aqua_rrs{wv}' and 'insitu_rrs{wv}' columns.
        - cut (pandas.Series): Boolean mask indicating valid data points.
        - std (float): Standard deviation of the difference.
        - rel_std (float): Relative standard deviation.

    """
    diff = modis[f'aqua_rrs{wv}'] - modis[f'insitu_rrs{wv}']
    cut = (np.abs(diff) < 100.) & np.isfinite(modis[f'aqua_rrs{wv}']) & (modis[f'aqua_rrs{wv}'] > 0.)
    # Sigma clip
    _, low, high = sigmaclip(diff[cut], low=sig_cut, high=sig_cut)
    sig_cut = (diff > low) & (diff < high)
    cut &= sig_cut
    #
    std = np.std(diff[cut])
    rel_std = np.std(np.abs(diff[cut])/modis[f'aqua_rrs{wv}'][cut])
    # Return
    return diff, cut, std, rel_std

def calc_errors():
    """
    Calculate errors for MODIS satellite data.

    Returns:
        dict: A dictionary containing the calculated errors for each wavelength.
              The keys are the wavelengths and the values are tuples containing
              the standard deviation and relative standard deviation.
    """
    # Load
    modis = load_matchups()

    err_dict = {}
    for wv in modis_wave:
        diff, cut, std, rel_std = calc_stats(modis, wv)
        #
        print(f'wv: {wv}, std={std:0.5f} sr^-1, rel_std={rel_std:0.2f}%')
        err_dict[wv] = (std, rel_std)

    # Return
    return err_dict
    