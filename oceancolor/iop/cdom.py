""" Methods related to CDOM """

import numpy as np

from scipy.optimize import curve_fit

from IPython import embed

def a_exp(wave:np.ndarray, S_CDOM:float=0.0176,
          wave0:float=440., const:float=0.):
    #  Return
    return np.exp(-S_CDOM * (wave-wave0)) + const # m^-1


def a_pow(wave:np.ndarray, S:float=-5.91476,
          wave0:float=440.):
    #  Return
    return (wave/wave0)**S

def fit_exp_norm(wave:np.ndarray, a_cdom:np.ndarray):

    def func(x, a440):
        return a440*a_exp(x)

    iwv = np.argmin(np.abs(wave-440.))
    p0 = [a_cdom[iwv]]
    return curve_fit(func, wave, a_cdom, p0=p0) #sigma=sig_y, #maxfev=maxfev)

def fit_exp_tot(wave:np.ndarray, a_cdom:np.ndarray):

    def func(x, a440, S, wave0):
        return a440*a_exp(x, S_CDOM=S, wave0=wave0)

    iwv = np.argmin(np.abs(wave-440.))
    p0 = [a_cdom[iwv], 0.0176, 440.]
    return curve_fit(func, wave, a_cdom, p0=p0) #sigma=sig_y, #maxfev=maxfev)


def fit_pow(wave:np.ndarray, a_cdom:np.ndarray):

    def func(x, a440, exponent, wave0=440.):
        return a440*(x/wave0)**exponent

    iwv = np.argmin(np.abs(wave-440.))
    p0 = [a_cdom[iwv], -6]
    return curve_fit(func, wave, a_cdom, p0=p0) #sigma=sig_y, #maxfev=maxfev)
