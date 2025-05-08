""" Cross-sections of ocean particles """
import numpy as np

import numpy as np


def detritus_abs(lamb:float):
    """ Absorption cross-section of detritus

    Taken from Stramski, Bricaud and Morel (2001)

    Args:
        lamb (float or np.ndarray): wavelength in nm

    Returns:
        float or np.ndarray: absorption cross-section in micron^-2
    """
    return 8.791e-4 * np.exp(-0.00847 * lamb)

def detritus_scatt(lamb:float):
    """ Scattering cross-section of detritus

    Taken from Stramski, Bricaud and Morel (2001)

    Args:
        lamb (float or np.ndarray): wavelength in nm

    Returns:
        float or np.ndarray: scattering cross-section in micron^-2
    """
    return 0.1425 * lamb**(-0.9445)

def detritus_backscatt(lamb:float):
    """ Backscattering cross-section of detritus

    Taken from Stramski, Bricaud and Morel (2001)

    Args:
        lamb (float or np.ndarray): wavelength in nm

    Returns:
        float or np.ndarray: backscattering cross-section in micron^-2
    """
    return 5.881e-4 * lamb**(-0.8997)

def minerals_abs(lamb:float):
    """ Absorption cross-section of minerals

    Taken from Stramski, Bricaud and Morel (2001)

    Args:
        lamb (float or np.ndarray): wavelength in nm

    Returns:
        float or np.ndarray: absorption cross-section in micron^-2
    """
    return 1.013e-3 * np.exp(-0.00846 * lamb)

def mineral_scatt(lamb:float):
    """ Scattering cross-section of minerals

    Taken from Stramski, Bricaud and Morel (2001)

    Args:
        lamb (float or np.ndarray): wavelength in nm

    Returns:
        float or np.ndarray: scattering cross-section in micron^-2
    """
    return 0.7712 * lamb**(-0.9764)

def mineral_backscatt(lamb:float):
    """ Backscattering cross-section of minearls

    Taken from Stramski, Bricaud and Morel (2001)

    Args:
        lamb (float or np.ndarray): wavelength in nm

    Returns:
        float or np.ndarray: backscattering cross-section in micron^-2
    """
    return 1.790e-2 * lamb**(-0.9140)

def bubbles_abs(lamb:float):
    """ Absorption cross-section of bubbles

    Taken from Stramski, Bricaud and Morel (2001)

    Args:
        lamb (float or np.ndarray): wavelength in nm

    Returns:
        float or np.ndarray: absorption cross-section in micron^-2
    """
    if isinstance(lamb, np.ndarray):
        return np.zeros_like(lamb)
    else:
        return 0.

def bubbles_scatt(lamb:float):
    """ Scattering cross-section of bubbles

    Taken from Stramski, Bricaud and Morel (2001)

    Args:
        lamb (float or np.ndarray): wavelength in nm

    Returns:
        float or np.ndarray: scattering cross-section in micron^-2
    """
    const = 4607.873
    if isinstance(lamb, np.ndarray):
        return const * np.ones_like(lamb)
    else:
        return const

def bubbles_backscatt(lamb:float):
    """ Backscattering cross-section of bubbles

    Taken from Stramski, Bricaud and Morel (2001)

    Args:
        lamb (float or np.ndarray): wavelength in nm

    Returns:
        float or np.ndarray: scattering cross-section in micron^-2
    """
    const = 55.359
    if isinstance(lamb, np.ndarray):
        return const * np.ones_like(lamb)
    else:
        return const
