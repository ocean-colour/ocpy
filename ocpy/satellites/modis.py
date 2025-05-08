""" Items related to MODIS """
import os
import numpy as np
from importlib.resources import files

from scipy.stats import sigmaclip

import pandas


from ocpy.satellites import utils as sat_utils


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

modis_wave = np.array([412, 443, 488, 531, 547, 
                       555, 667, 678])  # Narrow bands only
#modis_aqua_error = np.array([0.00141, 0.00113, 
#                    0.00113,  # Assumed for 469
#                    0.00113, 0.00102, 0.00117, 0.00120, 
#                    0.00070,  # Assumed for 645
#                    0.00056, 0.00060,
#                    0.00060,  # Assumed for 748
#                    ])

def load_matchups():
    """
    Load the MODIS matchups.

    Returns:
        pandas.DataFrame: DataFrame containing the MODIS matchups.
    """
    modis_file = files('ocpy').joinpath(
        os.path.join('data', 'satellites', 'MODIS_matchups_rrs.csv'))
    modis = pandas.read_csv(modis_file, comment='#')
    return modis


def calc_errors(rel_in_situ_error:float=None, reduce_by_in_situ:float=None, verbose:bool=False):
    """
    Calculate errors for MODIS satellite data.

    Args:
        rel_in_situ_error (float): The relative error in the in situ data. Default is 0.05.
        reduce_by_in_situ (bool): Whether to reduce the error by the in situ error. 
            If provided, reduce by this factor, e.g. sqrt(2)

    Returns:
        dict: A dictionary containing the calculated errors for each wavelength.
              The keys are the wavelengths and the values are tuples containing
              the standard deviation and relative standard deviation.
    """
    # Load
    modis = load_matchups()

    err_dict = {}

    for wv in modis_wave:
        diff, cut, std, rel_std = sat_utils.calc_stats(
            modis, wv, ['aqua_rrs', 'insitu_rrs'], rel_in_situ_error)
        # Reduce?
        if reduce_by_in_situ is not None:
            std /= reduce_by_in_situ
            rel_std /= reduce_by_in_situ
        #
        if verbose:
            print(f'wv: {wv}, std={std:0.5f} sr^-1, rel_std={rel_std:0.2f}%')
        err_dict[wv] = (std, rel_std)

    # Return
    return err_dict
    