""" Methods to estimate Chl-a from remote sensing reflectance. """

import numpy as np

def oc2(wave:np.ndarray, Rrs:np.ndarray):

    # Coeff
    a0 = 0.3410
    a1 = -3.0010
    a2 = 2.8110
    a3 = -2.0410
    a4 = -0.0400

    # Wavelengths
    i490 = np.argmin(np.abs(wave-490))
    i555 = np.argmin(np.abs(wave-555))

    # Max
    max_num = Rrs[i490]

    # Ratio me
    R = np.log10(max_num / Rrs[i555])

    # Finish
    Chl = 10**(a0 + a1*R + a2*R**2 + a3*R**3) + a4

    # Return
    return Chl

def oc4(wave:np.ndarray, Rrs:np.ndarray):

    # Coeff
    a0 = 0.4708
    a1 = -3.8469
    a2 = 4.5338
    a3 = -2.4434
    a4 = -0.0414

    # Wavelengths
    i443 = np.argmin(np.abs(wave-443))
    i490 = np.argmin(np.abs(wave-490))
    i510 = np.argmin(np.abs(wave-510))
    i555 = np.argmin(np.abs(wave-555))

    # Max
    max_num = np.max([Rrs[i443], Rrs[i490], Rrs[i510]])

    # Ratio me
    R = np.log10(max_num / Rrs[i555])

    # Finish
    Chl = 10**(a0 + a1*R + a2*R**2 + a3*R**3) + a4

    # Return
    return Chl