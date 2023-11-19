""" Pigment data and the like """
import numpy as np

from scipy.interpolate import interp1d

from oceancolor.ph import load_data as load_ph

def pg():
    peak_loc=[406,434,453,470,492,523,550,584,617,
          638,660,675]  
    FWHM = [37.68 , 28.26 , 28.26 , 30.615, 37.68 , 
        32.97 , 32.97 , 37.68 , 30.615, 25.905, 
        25.905, 23.55 ]

def a_chl(wave:np.ndarray, ctype:str='a',
          source:str='clementson2019'):

    # Load data
    if source == 'clementson2019':
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