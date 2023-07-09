""" Module to load polarization data """
import os
from pkg_resources import resource_filename

import numpy as np

import pandas

def koetner2021(sheet:str=None):
    """ Load the Koetner et al. 2021 VSF data

    Args:
        sheet (str, optional): Name of the sheet
            ['Lagoon', 'Mineral', 'BS', 'BS-uncorrected']
        build_vsf (bool, optional): 
            If True and a sheet was specified, load up
            the VSF data into a numpy array. Defaults to True.

    Returns:
        pandas.DataFrame or tuple: 
            DataFrame of the full table or tuple of the sheet, psis, vsf
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
