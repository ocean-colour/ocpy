""" Standard measures from Tara Oceans data."""

import numpy as np
import pandas

from matplotlib import pyplot as plt
import seaborn as sns

from oceancolor.tara import spectra

from IPython import embed

def chla_boss13(tara_tbl:pandas.DataFrame, debug:bool=False):

    # ap_676
    ap_676, _ = spectra.single_value(tara_tbl, 676., wv_delta=7., flavor='ap')

    # ap_650
    ap_650, _ = spectra.single_value(tara_tbl, 650., wv_delta=7., flavor='ap')

    # ap_715
    ap_715, _ = spectra.single_value(tara_tbl, 715., wv_delta=7., flavor='ap')

    # aphi_676
    aphi_676 = ap_676 - 39.*ap_650/65. - 26*ap_715/65.

    # Chla
    Chla = 157. * aphi_676**1.22  # mg/m^3
    neg = aphi_676 < 0.
    Chla[neg] = 0.

    # Add in place
    tara_tbl['Chla'] = Chla

    if debug:
        sns.histplot(np.maximum(Chla,1e-3), bins=100, log_scale=True)
        plt.show()
        embed(header='35 of measures.py')

    # Return
    return
    
def poc(tara_tbl:pandas.DataFrame, debug:bool=False):

    # pp_660
    cp_660, _ = spectra.single_value(tara_tbl, 660., wv_delta=7., flavor='cp')

    # POC
    POC = 380 * cp_660 # mg C / m^3

    # Add in place
    tara_tbl['POC'] = POC
    # Return
    return

def add_derived(tara_tbl:pandas.DataFrame, quantities:list=['all']):

    if 'all' in quantities:
        quantities = ['Chla', 'POC']

    # Chla
    if 'Chla' in quantities:
        chla_boss13(tara_tbl)

    # POC
    if 'POC' in quantities:
        poc(tara_tbl)

    # apxxx
    for item in quantities:
        if item.startswith('ap'):
            ap_xxx, _ = spectra.single_value(tara_tbl, float(item[2:]), wv_delta=7., flavor='ap')
            tara_tbl[item] = ap_xxx