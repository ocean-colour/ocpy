""" I/O for phytoplankton data. """

import os
from pkg_resources import resource_filename
import warnings

import scipy.io as sio

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
            'ocpy', 'data'), 'phytoplankton', 
            'stramski2001_table1.ascii')
    # Read
    s2001_tab1 = pandas.read_csv(s2001_tab1_file, sep='\t', 
                                 header=0)

    tables['table1'] = s2001_tab1

    # Stramski Absorption
    s2001_abs_file = os.path.join(resource_filename(
            'ocpy', 'data'), 'phytoplankton', 
            'Stramski 2001_absorption cross sections_18 species.xlsx')
    if os.path.isfile(s2001_abs_file):
        df_abs = pandas.read_excel(s2001_abs_file, sheet_name='Sheet1')
        tables['abs'] = df_abs
    else:
        warnings.warn('Stramski+ 2001 absorption cross sections not found. Request them')
        

    # Stramski Absorption
    s2001_att_file = os.path.join(resource_filename(
            'ocpy', 'data'), 'phytoplankton', 
            'Stramski 2001_attenuation cross sections_18 species.xlsx')
    if os.path.isfile(s2001_att_file):
        df_att = pandas.read_excel(s2001_att_file, sheet_name='Sheet1')
        tables['att'] = df_att
    else:
        warnings.warn('Stramski+ 2001 attenuation cross sections not found. Request them')

    # Return
    return tables

def clementson2019():

    dfs = []
    for ss in [2,3]:
        c2019_tab_file = os.path.join(resource_filename(
            'ocpy', 'data'), 'phytoplankton', 
            f'clementson2019_{ss}.txt')
        df = pandas.read_table(c2019_tab_file, delim_whitespace=True, header=0)
        # Sort on wavelength
        df.sort_values(by='wave', inplace=True)
        dfs.append(df)

    # Return
    return dfs[0], dfs[1]

def bricaud():
    """
    Load the Bricaud phytoplankton data from their 2004 paper.

    https://ui.adsabs.harvard.edu/abs/2004JGRC..10911010B/abstract

    Returns:
        pandas.DataFrame: The loaded data with columns 'wave' and other properties.
    """

    # 2004
    b2004_tab_file = os.path.join(resource_filename(
            'ocpy', 'data'), 'phytoplankton', 
            'Bricaud_2004.csv')
    df = pandas.read_csv(b2004_tab_file, comment='#')
    df.sort_values(by='wave', inplace=True)

    # Return
    return df

#df0, df1 = clementson2019()
#embed(header='63 of load_data.py')

def moore1995():
    """
    Taken from Moore et al. (1995) "Compartive ..."
        Marine Ecology Progress Series 116: 259-275
        doi.org/10.3354/meps116259

    Species are:
       'Pro SS120 9'
       'ProSS120 70'
       'Pro MED4 9'
       'Pro MED4 70'
       'SynWH8103'

    Returns:
        pandas.DataFrame: The loaded data with columns 'wave' and the various species
    """

    # Load
    moore_file = os.path.join(resource_filename(
            'ocpy', 'data'), 'phytoplankton', 
            'data_moore.mat')
    moore_data = sio.loadmat(moore_file)

    # Extract
    df = pandas.DataFrame()
    df['wave'] = moore_data['wvl'][0,:]
    # Spectra
    for ss in range(moore_data['spectra'].shape[1]):
        lbl = moore_data['labels'][0][ss][0]
        df[lbl] = moore_data['spectra'][:,ss]
    df.sort_values(by='wave', inplace=True)

    # Return
    return df