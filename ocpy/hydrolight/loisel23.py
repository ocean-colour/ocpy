""" Loisel+2023 Hydrolight Outputs """

import os
import numpy as np
import warnings

import xarray

from IPython import embed

if os.getenv('OS_COLOR') is not None:
    l23_path = os.path.join(os.getenv('OS_COLOR'),
                        'data', 'Loisel2023')
else:
    warnings.warn("OS_COLOR not set. Using current directory.")
    l23_path = './'                    

def load_ds(X:int, Y:int, profile:bool=False):
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
        profile (bool, optional): Flag to load the profile data. Defaults to False.


    Returns:
        xa.Dataset: dataset from L+23
    """
    ps = '_profile' if profile else ''

    # Load up the data
    variable_file = os.path.join(l23_path,
                                 f'Hydrolight{X}{Y:02d}{ps}.nc')
    ds = xarray.load_dataset(variable_file, engine='h5netcdf')

    # Return
    return ds

def calc_Chl(ds):
    """
    Calculate chlorophyll concentration from the given dataset.

    Args:
        ds (xarray.Dataset): Dataset object containing the necessary data.

    Returns:
        np.ndarray: Chlorophyll concentration calculated from the dataset.
    """
    i440 = np.argmin(np.abs(ds.Lambda.data - 440.))
    Chl = ds.aph.data[:,i440] / 0.05582

    # Return
    return Chl
