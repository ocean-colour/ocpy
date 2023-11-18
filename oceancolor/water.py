""" Water methods """

import os
from importlib import resources

import numpy as np
from scipy.interpolate import interp1d

import pandas

def load_rsr_gsfc():
    """ Load the GSFC RSR table for water

    Returns:
        pandas.DataFrame: table of data with columns
            wavelength (nm), a_w, b_w
    """
    # File
    gsfc_file = os.path.join(
        resources.files('oceancolor'), 
        'data', 'water', 'water_coef.txt')

    # Load table, ignore #
    df = pandas.read_csv(gsfc_file, sep=' ', comment='#')

    # Return
    return df

def a_water(wv:np.ndarray):
    # Load
    df_water = load_rsr_gsfc()

    # Interpolate
    f = interp1d(df_water['wavelength'],
                        df_water['aw'], bounds_error=False, 
                        fill_value=np.nan)
    # Done
    return f(wv)