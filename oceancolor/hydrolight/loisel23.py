""" Loisel+2023 Hydrolight Outputs """

import os
import xarray
import numpy as np

from oceancolor.tara import io as tara_io 

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
