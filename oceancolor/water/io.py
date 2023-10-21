""" I/O for Water """

import os
from importlib import resources

import pandas

def load_rsr_gsfc():
    # File
    gsfc_file = os.path.join(
        resources.files('oceancolor'), 
        'data', 'water', 'water_coef.txt')

    # Load table, ignore #
    df = pandas.read_csv(gsfc_file, sep=' ', comment='#')

    # Return
    return df