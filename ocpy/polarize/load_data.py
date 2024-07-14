""" Module to load polarization data """
import os
from pkg_resources import resource_filename

import numpy as np

import pandas

def koetner2020(sheet:str=None):
    """ Load Koetner et al. 2020 from Figure 4

    Args:
        sheet (str, optional): Name of the sheet
            ['Header', 'P12', 'P22', 'P12 2018 corrections', 'P22 no correction']

    Returns:
        pandas.DataFrame or tuple: 
            DataFrame of the full table or tuple of the sheet, psis [deg], vsf [1/m sr]
    """
    file_2020 = os.path.join(resource_filename('oceancolor', 'data'), 'polarization',
                         'Koestner_et_al_2020_AO_Fig4.xlsx')
    # Open                        
    k2020 = pandas.read_excel(file_2020, sheet_name=None)

    # Sheet?
    if sheet is None:
        return k2020

    sheet = k2020[sheet]

    # Parse
    psis = []
    psi_cols = []
    cols = []
    for key in sheet.keys():
        try:
            psis.append(float(key))
        except:
            pass
        else:
            psi_cols.append(key)
            cols.append(sheet[key])
                        
    # Recast
    psis = np.array(psis)
    data = np.concatenate(cols).reshape(len(psi_cols),(len(sheet)))

    # Chop up
    samples = data[:,0:15]
    median = data[:,15]
    mean = data[:,16]

    return sheet, psis, samples, median, mean 

def koetner2021(sheet:str=None):
    """ Load the Koetner et al. 2021 VSF data

    Args:
        sheet (str, optional): Name of the sheet
            ['Lagoon', 'Mineral', 'BS', 'BS-uncorrected']

    Returns:
        pandas.DataFrame or tuple: 
            DataFrame of the full table or tuple of the sheet, psis [deg], vsf [1/m sr]
    """
    file_2021 = os.path.join(resource_filename('oceancolor', 'data'), 'polarization',
                         'Koestner-et-al-2021_AO_VSFs.xlsx')
    # Open                        
    k2021 = pandas.read_excel(file_2021, sheet_name=None)

    # Rename BS for convenience
    k2021['BS'] = k2021.pop('Beaufort Sea - corrected')
    k2021['BS-uncorrected'] = k2021.pop('Beaufort Sea - uncorrected')

    # Sheet?
    if sheet is None:
        return k2021

    sheet = k2021[sheet]

    # Parse
    psis = []
    psi_cols = []
    cols = []
    for key in sheet.keys():
        try:
            psis.append(float(key))
        except:
            pass
        else:
            psi_cols.append(key)
            cols.append(sheet[key])
                        
    # Recast
    psis = np.array(psis)
    vsf = np.concatenate(cols).reshape(len(psi_cols),(len(sheet)))

    # Return
    return sheet, psis, vsf
