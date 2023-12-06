""" Loisel+2023 Hydrolight Outputs """

import os
import xarray
import numpy as np

from oceancolor.tara import io as tara_io 
from oceancolor.tara import spectra
from oceancolor import water
from oceancolor.utils import spectra as spec_utils 

l23_path = os.path.join(os.getenv('OS_COLOR'),
                        'data', 'Loisel2023')

def load_ds(X:int, Y:int):
    """ Load up the data

    Data may be downloaded from:
    https://datadryad.org/stash/dataset/doi:10.6076/D1630T

    Args:
        X (int): simulation scenario   
            X = 1: No inelastic processes included.
            X = 2: Raman scattering by water molecules included.
            X = 4: Raman scattering by water molecules and fluorescence of chlorophyll-a included.
        Y (int):  solar zenith angle used in the simulation, and 
            represents a value of 00, 30, or 60 degrees.


    Returns:
        xa.Dataset: dataset from L+23
    """

    # Load up the data
    variable_file = os.path.join(l23_path,
                                 f'Hydrolight{X}{Y:02d}.nc')
    ds = xarray.load_dataset(variable_file)

    # Return
    return ds

def tara_matched_to_l23(low_cut:float=400., high_cut:float=705., X:int=4, Y:int=0):
    """ Generate Tara spectra matched to L23

    Restricted on wavelength and time of cruise
    as per Patrick Gray recommendations

    Args:
        low_cut (float, optional): low cut wavelength. Defaults to 400..
        high_cut (float, optional): high cut wavelength. Defaults to 705..
        X (int, optional): simulation scenario. Defaults to 4.
        Y (int, optional): solar zenith angle used in the simulation.
            Defaults to 0.

    Returns:
        tuple: wavelength grid, Tara spectra, L23 spectra
    """

    # Load up the data
    l23_ds = load_ds(X, Y)

    # Load up Tara
    print("Loading Tara..")
    tara_db = tara_io.load_tara_db(clean_dates=True)

    # Spectra
    wv_nm, all_a_p, all_a_p_sig = spectra.spectra_from_table(tara_db)

    # Wavelengths, restricted to > 400 nm
    cut = (l23_ds.Lambda > low_cut) & (l23_ds.Lambda < high_cut)
    l23_a = l23_ds.a.data[:,cut]

    # Rebin
    wv_grid = l23_ds.Lambda.data[cut]
    tara_wv = np.append(wv_grid, [high_cut+5.]) - 2.5 # Because the rebinning is not interpolation
    rwv_nm, r_ap, r_sig = spectra.rebin_to_grid(
        wv_nm, all_a_p, all_a_p_sig, tara_wv)

    # Add in water
    print("Adding in water..")
    df_water = water.load_rsr_gsfc()
    a_w, _ =  spec_utils.rebin(df_water.wavelength.values, 
                        df_water.aw.values, np.zeros_like(df_water.aw),
                        wv_grid)

    tara_a_water = r_ap+np.outer(np.ones(r_ap.shape[0]), a_w)

    # Polish Tara for PCA
    bad = np.isnan(tara_a_water) | (tara_a_water <= 0.)
    ibad = np.where(bad)

    mask = np.ones(tara_a_water.shape[0], dtype=bool)
    mask[np.unique(ibad[0])] = False

    # Cut down: Aggressive but necessary
    tara_a_water = tara_a_water[mask,:]


    # Return
    return wv_grid, tara_a_water, l23_a