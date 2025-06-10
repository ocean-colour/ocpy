""" Methods related to Kd """

import numpy as np


def calc_kd_lee(a:np.ndarray, bb:np.ndarray, bbw:np.ndarray, theta_sun:float=0.):

    # Calculate the diffuse attenuation coefficient Kd using the Lee et al. (1998) method.
    Kd = (1. + 1.005*theta_sun) * a + 4.259*(1-0.265*bbw/bb) * (
        1-0.52*np.exp(-10.8*a))

    # Ensure Kd is non-negative
    Kd = np.maximum(Kd, 0)

    # Return
    return Kd