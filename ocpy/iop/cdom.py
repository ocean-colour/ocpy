""" Methods related to CDOM """

import numpy as np

from scipy.optimize import curve_fit

from IPython import embed

def a_exp(wave:np.ndarray, S_CDOM:float=0.0176,
        wave0:float=440., const:float=0.):
    """
    Calculate the absorption coefficient of colored dissolved organic matter (CDOM) using the exponential function.

    Args:
        wave (np.ndarray): Array of wavelengths.
        S_CDOM (float, optional): CDOM specific absorption coefficient [m^-1/(mg/L)].
        wave0 (float, optional): Reference wavelength [nm].
        const (float, optional): Constant term added to the exponential function.

    Returns:
        np.ndarray: Array of absorption values [m^-1].
    """
    return np.exp(-S_CDOM * (wave-wave0)) + const


def a_pow(wave:np.ndarray, S:float=-5.91476,
        wave0:float=440.):
    """
    Calculate the absorption coefficient power law.

    Parameters:
        wave (np.ndarray): Array of wavelengths.
        S (float): Power law exponent (default: -5.91476).
        wave0 (float): Reference wavelength (default: 440).

    Returns:
        np.ndarray: Array of absorption coefficients calculated using the power law formula.
    """
    return (wave/wave0)**S

def fit_exp_norm(wave:np.ndarray, a_cdom:np.ndarray):
    """
    Fits an exponential function to the given wavelength and absorption coefficient data.

    Only fits the normalization; S_CDOM is fixed at its default value

    Args:
        wave (np.ndarray): Array of wavelengths.
        a_cdom (np.ndarray): Array of absorption coefficients.

    Returns:
        tuple: A tuple containing the optimized parameters and covariance matrix of the fit.
    """
    def func(x, a440):
        return a440*a_exp(x)

    iwv = np.argmin(np.abs(wave-440.))
    p0 = [a_cdom[iwv]]
    return curve_fit(func, wave, a_cdom, p0=p0) #sigma=sig_y, #maxfev=maxfev)

def fit_exp_tot(wave:np.ndarray, a_cdom:np.ndarray):
    """ Fit an exponential function to the given wavelength 
    and absorption coefficient data.

    Fitted parameters are: a440, S_CDOM

    Args:
        wave (np.ndarray): Wavelengths
        a_cdom (np.ndarray): Absorption coefficients

    Returns:
        tuple: A tuple containing the optimized parameters and covariance matrix of the fit.
    """

    def func(x, a440, S):#, wave0):
        return a440*a_exp(x, S_CDOM=S)#, wave0=wave0)

    iwv = np.argmin(np.abs(wave-440.))
    p0 = [a_cdom[iwv], 0.0176]
    return curve_fit(func, wave, a_cdom, p0=p0) #sigma=sig_y, #maxfev=maxfev)


def fit_pow(wave:np.ndarray, a_cdom:np.ndarray):
    """
    Fits a power-law function to the given wavelength and absorption coefficient data.

    Parameters:
        wave (np.ndarray): Array of wavelengths.
        a_cdom (np.ndarray): Array of absorption coefficients.

    Returns:
        tuple: A tuple containing the optimized parameters and covariance matrix of the fit.
    """

    def func(x, a440, exponent, wave0=440.):
        return a440*(x/wave0)**exponent

    iwv = np.argmin(np.abs(wave-440.))
    p0 = [a_cdom[iwv], -6]
    return curve_fit(func, wave, a_cdom, p0=p0) #sigma=sig_y, #maxfev=maxfev)
