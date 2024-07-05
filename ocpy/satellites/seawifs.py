""" Items related to SeaWiFS """
import os
import numpy as np
from importlib.resources import files

from scipy.stats import sigmaclip

import pandas

from oceancolor.satellites import utils as sat_utils


# https://oceancolor.gsfc.nasa.gov/resources/atbd/rrs/#sec_4


# Calculate with the code below using sigclip
#wv: 412, std=0.00143 sr^-1, rel_std=34.72%
#wv: 443, std=0.00114 sr^-1, rel_std=36.72%
#wv: 490, std=0.00091 sr^-1, rel_std=4.28%
#wv: 510, std=0.00063 sr^-1, rel_std=2.11%
#wv: 555, std=0.00071 sr^-1, rel_std=1.77%
#wv: 670, std=0.00026 sr^-1, rel_std=99.60%

seawifs_wave = np.array([412, 443, 490, 510, 555, 670])
#seawifs_error = np.array([0.00143, 0.00114, 0.00091, 
#                     0.00063, 0.00071, 0.00026])

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



def calc_errors(rel_in_situ_error=0.05):
    """
    Calculate errors for SeaWiFS data.

    Args:
        rel_in_situ_error (float): The relative error in the in situ data. Default is 0.05.

    Returns:
        dict: A dictionary containing the standard deviation and relative standard deviation
              for each wavelength in the SeaWiFS data.
    """
    # Load
    seawifs = load_matchups()

    err_dict = {}
    for wv in seawifs_wave:
        diff, cut, std, rel_std = sat_utils.calc_stats(
            seawifs, wv, ['seawifs_rrs', 'insitu_rrs'], rel_in_situ_error)
        #
        print(f'wv: {wv}, std={std:0.5f} sr^-1, rel_std={100*rel_std:0.2f}%')
        err_dict[wv] = (std, rel_std)

    # Return
    return err_dict