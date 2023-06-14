""" Cross-sections of ocean particles """
import numpy as np

def detritus_abs(lamb:float):
    """ Absorption cross-section of detritus

    Args:
        lamb (float or np.ndarray): wavelength in nm

    Returns:
        float or np.ndarray: absorption cross-section in micron^-2
    """
    return 8.791e-4 * np.exp(-0.00847 * lamb)

def minerals_abs(lamb:float):
    """ Absorption cross-section of minerals

    Args:
        lamb (float or np.ndarray): wavelength in nm

    Returns:
        float or np.ndarray: absorption cross-section in micron^-2
    """
    return 1.013-3 * np.exp(-0.00846 * lamb)

def bubbles_abs(lamb:float):
    """ Absorption cross-section of bubbles

    Args:
        lamb (float or np.ndarray): wavelength in nm

    Returns:
        float or np.ndarray: absorption cross-section in micron^-2
    """
    if isinstance(lamb, np.ndarray):
        return np.zeros_like(lamb)
    else:
        return 0.