""" Methods for PACE """

from importlib.resources import files
import os
import numpy as np

import pandas

from IPython import embed

# Defatuls
PACE_wave = np.arange(400, 701, 5)

def gen_noise_vector(wave:np.ndarray, include_sampling:bool=False):
    """
    Generate a noise vector based on PACE error.

    Parameters:
        wave (np.ndarray): Array of wavelengths.

    Returns:
        np.ndarray: Noise vector based on PACE error.
    """
    # Load PACE error
    pace_file = files('boring').joinpath(os.path.join('data', 'PACE', 'PACE_error.csv'))
    PACE_errors = pandas.read_csv(pace_file)

    # Interpolate
    PACE_error = np.interp(wave, PACE_errors['wave'], PACE_errors['PACE_sig'])
    dwv = np.abs(np.median(np.roll(wave, -1) - wave))
    dwv_PACE = np.abs(np.median(np.roll(PACE_errors['wave'], -1) - PACE_errors['wave']))

    # Correct for sampling (approx)
    if include_sampling:
        PACE_error /= np.sqrt(dwv/dwv_PACE)

    # Return
    return PACE_error