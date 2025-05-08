""" Methods related to the absorption of phytoplankton. """

import os
from pkg_resources import resource_filename

import pandas

def load_bricaud1998():
    """
    Load the Bricaud 1998 aph data.

    Returns:
        pandas.DataFrame: The loaded Bricaud 1998 aph data.
    """
    b1998_tab_file = os.path.join(resource_filename(
            'ocpy', 'data'), 'phytoplankton', 
            'aph_bricaud_1998.txt')

    b1998_tab = pandas.read_csv(b1998_tab_file, comment='#')

    # Return
    return b1998_tab
