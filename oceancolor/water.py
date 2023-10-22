""" I/O for Water """

import os
from importlib import resources

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