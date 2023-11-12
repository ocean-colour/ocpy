""" I/O for IOP data. """

import os
from importlib import resources

import pandas

from IPython import embed

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

def load_IOCCG_2018():
    """ Load the IOCCG 2018 pure water model

    From Rick Reynolds

    Returns:
        pandas.DataFrame: table of data with columns
            wavelength (nm), a_w, b_w
    """
    # File
    data_file = os.path.join(
        resources.files('oceancolor'), 
        'data', 'water', 'a_water_IOCCG_2018.csv')

    # Load table, ignore #
    df = pandas.read_csv(data_file, comment='#')

    # Chop down
    df = df[df.wavelength < 2000.]

    # Return
    return df[['wavelength', 'aw']]