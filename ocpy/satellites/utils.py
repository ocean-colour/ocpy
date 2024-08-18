""" Utilities for satellite data """

import numpy as np
from scipy.stats import sigmaclip

from IPython import embed


def calc_stats(tbl, wv, key_roots:list, 
               rel_in_situ_error:float=None):
    """
    Calculate statistics for a given table and wavelength.

    Args:
        tbl (pandas.DataFrame): The table containing the data.
        wv (str): The wavelength.
        key_roots (list): A list of key roots used to access the data columns.
            satellite, in_situ
        rel_in_situ_error (float, optional): 
            The relative in-situ error to subtract off.
            e.g. 0.5 means 1/2 the error is due to in-situ

    Returns:
        tuple: A tuple containing the following elements:
            - diff (numpy.ndarray): The difference between two data columns.
            - cut (numpy.ndarray): A boolean mask indicating valid data points.
            - std (float): The standard deviation of the difference.
            - rel_std (float): The relative standard deviation of the difference.

    """
    diff = tbl[f'{key_roots[0]}{wv}'] - tbl[f'{key_roots[1]}{wv}']
    cut = (np.abs(diff) < 100.) & np.isfinite(tbl[f'{key_roots[0]}{wv}']) & (tbl[f'{key_roots[0]}{wv}'] > 0.)
    # Clip
    _, low, high = sigmaclip(diff[cut], low=3., high=3.)
    # Cut
    sig_cut = (diff > low) & (diff < high)
    cut &= sig_cut
    # Calculate
    std = np.std(diff[cut])
    rel_std = np.std(np.abs(diff[cut])/tbl[f'{key_roots[0]}{wv}'][cut])

    # Subtract in situ error?
    if rel_in_situ_error is None:
        return diff, cut, std, rel_std

    sig_insitu = tbl[f'{key_roots[1]}{wv}'] * rel_in_situ_error
    med_sig_insitu = np.median(sig_insitu)
    print(f"wv: {wv}, sig_insitu: {med_sig_insitu}, std: {std}, rel_std: {rel_std}")

    if med_sig_insitu > std:
        embed(header='41 of utils')

    if wv == 678:
        embed(header='47 of utils')
    
    std = np.sqrt(std**2 - med_sig_insitu**2)
    rel_std = np.sqrt(rel_std**2 - np.median(sig_insitu/tbl[f'{key_roots[0]}{wv}'])**2)
    # Finish
    return diff, cut, std, rel_std