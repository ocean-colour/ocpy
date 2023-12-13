""" Pigment data and the like """
import numpy as np

from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from oceancolor.ph import load_data as load_ph

def pg():
    peak_loc=[406,434,453,470,492,523,550,584,617,
          638,660,675]  
    FWHM = [37.68 , 28.26 , 28.26 , 30.615, 37.68 , 
        32.97 , 32.97 , 37.68 , 30.615, 25.905, 
        25.905, 23.55 ]

def a_chl(wave:np.ndarray, ctype:str='a',
          source:str='bricaud'):
    """
    Load up the chlorophyll absorption coefficient.

    Args:
        wave (np.ndarray): Array of wavelengths.
        ctype (str, optional): Chlorophyll type. Defaults to 'a'.
            Options are: 'a' (default), 'b', 'c12'
        source (str, optional): Data source. Defaults to 'bricaud'.

    Returns:
        np.ndarray: Array of chlorophyll absorption coefficients.
    """
    # Load data
    if source == 'clementson':
        _, df = load_ph.clementson2019()
    elif source == 'bricaud':
        df = load_ph.bricaud()
    else:
        raise IOError(f"Bad input source {source}")

    # Type
    key = f'Chl-{ctype}'

    # Interpolate
    f = interp1d(df.wave, df[key], kind='linear',
                 bounds_error=False, fill_value=0.)

    # Rinish
    return f(wave)


def fit_a_chl(wave:np.ndarray, a_ph:np.ndarray):
    """
    Fits Chl pigments to a presumed a_chl spectrum.

    Args:
        wave (np.ndarray): Array of wavelengths.
            These can be restricted to a subset
        a_ph (np.ndarray): Array of absorption coefficients.

    Returns:
        tuple: A tuple containing the optimized parameters and covariance matrix of the fit.
    """
    chla = a_chl(wave, ctype='a')
    chlb = a_chl(wave, ctype='b')
    chlc = a_chl(wave, ctype='c12')

    nrm_wv = [673.,440.,440.]
    p0 = []
    for wv, pig in zip(nrm_wv, [chla, chlb, chlc]):
        iwv = np.argmin(np.abs(wave-wv))
        nrm = pig[iwv]/a_ph[iwv]
        p0.append(1./nrm)

    # Func
    def func(x, Aa, Ab, Ac):
        return Aa*chla + Ab*chlb + Ac*chlc 

    # Fit
    return curve_fit(func, wave, a_ph, p0=p0) #sigma=sig_y, #maxfev=maxfev)
