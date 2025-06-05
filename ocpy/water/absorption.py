""" Water methods """

import os
from importlib import resources

import numpy as np
from scipy.interpolate import interp1d

import pandas

def load_rsr_gsfc():
    """ Load the GSFC RSR table for water

    Pope and Fry (1997) data from GSFC
    No salt

    Returns:
        pandas.DataFrame: table of data with columns
            wavelength (nm), a_w, b_w
    """
    # File
    gsfc_file = os.path.join(
        resources.files('ocpy'), 
        'data', 'water', 'water_coef.txt')

    # Load table, ignore #
    df = pandas.read_csv(gsfc_file, sep=' ', comment='#')

    # Return
    return df

def load_ioccg_2018():
    """
    Load the IOCCG 2018 water absorption data.

    Kindly provided by R. Reynolds (Scripps)

    Returns:
        pandas.DataFrame: The loaded data as a pandas DataFrame.
    """
    ioccg_file = os.path.join(
        resources.files('ocpy'), 
        'data', 'water', 'a_water_IOCCG_2018.csv')
    # Load table, ignore #
    df = pandas.read_csv(ioccg_file, comment='#')
    # Return
    return df

def a_water(wv:np.ndarray, data='GSFC'):
    """
    Calculate the absorption coefficient of water for the given wavelengths.

    Args:
        wv (np.ndarray): Array of wavelengths.
        data (str, optional): The data source to use. Defaults to 'GSFC'.

    Returns:
        np.ndarray: Array of absorption coefficients.
    """
    # Load
    if data == 'GSFC':
        df_water = load_rsr_gsfc()
    elif data == 'IOCCG': 
        df_water = load_ioccg_2018()
    else:
        raise ValueError(f"Unknown data: {data}")

    # Interpolate
    f = interp1d(df_water['wavelength'],
                        df_water['aw'], bounds_error=False, 
                        fill_value=np.nan)
    # Done
    return f(wv)
