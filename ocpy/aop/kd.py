""" Methods related to Kd """

import numpy as np


def calc_kd_lee(a:np.ndarray, bb:np.ndarray, bbw:np.ndarray, 
                theta_sun:float=0.):
    """
    Calculate the diffuse attenuation coefficient (Kd) using the Lee et al. (1998) method.

    Parameters:
    -----------
    a : np.ndarray
        Absorption coefficient (m^-1). This is a numpy array representing the absorption 
        properties of the water.
    bb : np.ndarray
        Backscattering coefficient (m^-1). This is a numpy array representing the 
        backscattering properties of the water.
    bbw : np.ndarray
        Backscattering coefficient of pure water (m^-1). This is a numpy array representing 
        the backscattering properties of pure water.
    theta_sun : float, optional
        Solar zenith angle in radians. Default is 0 (sun directly overhead).

    Returns:
    --------
    np.ndarray
        The diffuse attenuation coefficient (Kd) as a numpy array. Values are ensured to 
        be non-negative.

    Notes:
    ------
    The calculation is based on the Lee et al. (2013) method, which models the diffuse 
    attenuation coefficient as a function of absorption and backscattering properties 
    of water, as well as the solar zenith angle.
    """

    # Calculate the diffuse attenuation coefficient Kd using the Lee et al. (1998) method.
    Kd = (1. + 1.005*theta_sun) * a + 4.259*(1-0.265*bbw/bb) * (
        1-0.52*np.exp(-10.8*a))*bb

    # Ensure Kd is non-negative
    Kd = np.maximum(Kd, 0)

    # Return
    return Kd