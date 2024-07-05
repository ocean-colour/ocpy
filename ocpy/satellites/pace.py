""" Methods for PACE """

from importlib.resources import files
import os
import numpy as np

import pandas

from IPython import embed

def gen_noise_vector(wave:np.ndarray, include_sampling:bool=False):
    """
    Generate a noise vector based on PACE error.

    Parameters:
        wave (np.ndarray): Array of wavelengths.
        include_sampling (bool): Whether to adjust the error due
            to differences in sampling. Default is False.

    Returns:
        np.ndarray: Noise vector based on PACE error.
    """
    # Load PACE error
    pace_file = files('oceancolor').joinpath(os.path.join(
        'data', 'satellites', 'PACE_error.csv'))
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