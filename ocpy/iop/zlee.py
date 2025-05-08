""" Methods related to Zheng Lee's work. """

import numpy as np

def Y_from_Rrs(wave:np.ndarray, Rrs:np.ndarray):

    A_Rrs, B_Rrs = 0.52, 1.7

    rrs = Rrs / (A_Rrs + B_Rrs*Rrs)

    i440 = np.argmin(np.abs(wave-440))
    i555 = np.argmin(np.abs(wave-555))

    Y = 2.2 * (1 - 1.2 * np.exp(-0.9 * rrs[i440]/rrs[i555]))

    return Y