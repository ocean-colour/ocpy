""" I/O for phytoplankton data. """

import os
from pkg_resources import resource_filename

import pandas

from IPython import embed

def load_tables():
    """ Load the phytoplankton tables. 

    Returns:
        dict: dictionary of tables
    """
    # Stramski Table 1
    s2001_tab1_file = os.path.join(resource_filename(
            'oceancolor', 'data'), 'phytoplankton', 
            'stramski2001_table1.ascii')
    # Read
    s2001_tab1 = pandas.read_csv(s2001_tab1_file, sep='\t', 
                                 header=0)
    embed(header='io 22')


    # Return
    return s2001_tab1