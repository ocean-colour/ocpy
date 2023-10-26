""" I/O for phytoplankton data. """

import os
from pkg_resources import resource_filename
import warnings

import pandas

from IPython import embed

def stramski2001():
    """ Load data from Stramski et al. 2001

    Returns:
        dict: dictionary of tables
    """
    tables = {}

    # Stramski Table 1
    s2001_tab1_file = os.path.join(resource_filename(
            'oceancolor', 'data'), 'phytoplankton', 
            'stramski2001_table1.ascii')
    # Read
    s2001_tab1 = pandas.read_csv(s2001_tab1_file, sep='\t', 
                                 header=0)

    tables['table1'] = s2001_tab1

    # Stramski Absorption
    s2001_abs_file = os.path.join(resource_filename(
            'oceancolor', 'data'), 'phytoplankton', 
            'Stramski 2001_absorption cross sections_18 species.xlsx')
    if os.path.isfile(s2001_abs_file):
        df_abs = pandas.read_excel(s2001_abs_file, sheet_name='Sheet1')
        tables['abs'] = df_abs
    else:
        warnings.warn('Stramski+ 2001 absorption cross sections not found. Request them')
        

    # Stramski Absorption
    s2001_att_file = os.path.join(resource_filename(
            'oceancolor', 'data'), 'phytoplankton', 
            'Stramski 2001_attenuation cross sections_18 species.xlsx')
    if os.path.isfile(s2001_att_file):
        df_att = pandas.read_excel(s2001_att_file, sheet_name='Sheet1')
        tables['att'] = df_att
    else:
        warnings.warn('Stramski+ 2001 attenuation cross sections not found. Request them')

    # Return
    return tables