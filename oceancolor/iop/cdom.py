""" Methods related to CDOM """

import numpy as np

def a_exp(wave:np.ndarray, S_CDOM:float=0.0176,
          wave0:float=440., const:float=0.):
    #  Return
    return np.exp(-S_CDOM * (wave-wave0)) + const # m^-1


def a_pow(wave:np.ndarray, S:float=-5.91476,
          wave0:float=440.):
    #  Return
    return (wave/wave0)**S