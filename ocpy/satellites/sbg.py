""" Methods for SBG """

from importlib.resources import files
import os
import numpy as np

import pandas

from ocpy.satellites import pace as sat_pace

from IPython import embed

def gen_noise_vector(wave:np.ndarray, 
                     pix_size:float=300.):
    """
    Generate a noise vector based on PACE error.

    Parameters:
        wave (np.ndarray): Array of wavelengths.
        pix_size (float): Size of the pixel in meters. Default is 300.
            This is used to calculate the error relative to PACE

    Returns:
        np.ndarray: Noise vector based on PACE error.
    """
    # Load PACE error
    pace_error = sat_pace.gen_noise_vector(wave)

    # Degrade by pixel size
    boost = 1000. / pix_size
    sbg_error = pace_error * boost

    # Return noise vector
    return sbg_error