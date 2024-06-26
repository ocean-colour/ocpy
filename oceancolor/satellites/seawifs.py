""" Items related to SeaWiFS """
import os
import numpy as np
from importlib.resources import files

from scipy.stats import sigmaclip

import pandas


# https://oceancolor.gsfc.nasa.gov/resources/atbd/rrs/#sec_4


# Calculate with the code below using sigclip
#wv: 412, std=0.00143 sr^-1, rel_std=34.72%
#wv: 443, std=0.00114 sr^-1, rel_std=36.72%
#wv: 490, std=0.00091 sr^-1, rel_std=4.28%
#wv: 510, std=0.00063 sr^-1, rel_std=2.11%
#wv: 555, std=0.00071 sr^-1, rel_std=1.77%
#wv: 670, std=0.00026 sr^-1, rel_std=99.60%

seawifs_wave = np.array([412, 443, 490, 510, 555, 670])
seawifs_error = np.array([0.00143, 0.00114, 0.00091, 
                     0.00063, 0.00071, 0.00026])

def load_matchups():
    """
    Load the SeaWiFS matchups.

    Returns:
        pandas.DataFrame: DataFrame containing the SeaWiFS matchups.
    """
    seawifs_file = files('oceancolor').joinpath(os.path.join(
        'data', 'satellites', 'SeaWiFS_rrs_seabass.csv'))
    seawifs = pandas.read_csv(seawifs_file, comment='#')
    return seawifs

def calc_stats(tbl, wv):
    diff = tbl[f'seawifs_rrs{wv}'] - tbl[f'insitu_rrs{wv}']
    cut = (np.abs(diff) < 100.) & np.isfinite(tbl[f'seawifs_rrs{wv}']) & (tbl[f'seawifs_rrs{wv}'] > 0.)
    # Sigma clip
    _, low, high = sigmaclip(diff[cut], low=3., high=3.)
    sig_cut = (diff > low) & (diff < high)
    cut &= sig_cut
    #
    std = np.std(diff[cut])
    rel_std = np.std(np.abs(diff[cut])/tbl[f'seawifs_rrs{wv}'][cut])
    # Return
    return diff, cut, std, rel_std

def calc_errors():
    """
    Calculate errors for SeaWiFS data.

    Returns:
        dict: A dictionary containing the standard deviation and relative standard deviation
              for each wavelength in the SeaWiFS data.
    """
    # Load
    seawifs = load_matchups()

    err_dict = {}
    for wv in seawifs_wave:
        diff, cut, std, rel_std = calc_stats(seawifs, wv)
        #
        print(f'wv: {wv}, std={std:0.5f} sr^-1, rel_std={rel_std:0.2f}%')
        err_dict[wv] = (std, rel_std)

    # Return
    return err_dict