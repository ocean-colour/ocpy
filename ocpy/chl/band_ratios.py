""" Methods to estimate Chl-a from remote sensing reflectance.

Two generations of the OC4 maximum-band-ratio algorithm live here, and they are
**not** interchangeable:

* :func:`oc4` is the first OC4 (O'Reilly et al. 1998), a *modified cubic
  polynomial* -- a cubic in the log band ratio plus an additive constant
  outside the power of ten.  It is kept unchanged because BING and IOPtics
  (``ioptics.prep``) call it.
* :func:`oc4v4` is OC4 version 4 (O'Reilly et al. 2000), a quartic with no
  additive term.  This is the algorithm LS2 (Loisel et al. 2018) cites for its
  chlorophyll, so it is what an LS2 ``b_p`` side-chain should use.

On the authors' LS2 reference spectra the 1998 form reads 0.1-2.3% lower.
"""

import numpy as np

#: OC4 version 4 coefficients, highest power last (O'Reilly et al. 2000,
#: SeaWiFS Postlaunch Technical Report Series Vol. 11, Eq. 4).
OC4V4_COEFFS = (0.366, -3.067, 1.930, 0.649, -1.532)

#: SeaWiFS bands of the OC4 maximum band ratio [nm]: three blue numerators
#: and the green denominator.
OC4_BANDS = (443., 490., 510., 555.)


def oc2(wave:np.ndarray, Rrs:np.ndarray):
    """OC2 (O'Reilly et al. 1998): Chl from the Rrs(490)/Rrs(555) ratio.

    A modified cubic polynomial, ``10**(a0 + a1 R + a2 R^2 + a3 R^3) + a4``
    with ``R = log10(Rrs(490)/Rrs(555))``.  The nearest available band is used
    for each wavelength.

    Parameters
    ----------
    wave : numpy.ndarray
        Wavelengths [nm], shape ``(L,)``.
    Rrs : numpy.ndarray
        Remote-sensing reflectance [sr^-1], shape ``(L,)``.

    Returns
    -------
    float
        Chlorophyll-a concentration [mg m^-3].
    """

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
    """OC4 in its first, 1998 form: a cubic plus an additive constant.

    ``Chl = 10**(a0 + a1 R + a2 R^2 + a3 R^3) + a4`` with
    ``R = log10(max(Rrs(443), Rrs(490), Rrs(510)) / Rrs(555))`` and
    ``(a0..a4) = (0.4708, -3.8469, 4.5338, -2.4434, -0.0414)``.  This is the
    "modified cubic polynomial" that O'Reilly et al. (2000) describe as the
    first version of OC4 (O'Reilly et al. 1998); the coefficients are as
    carried in ocpy.  It is **not** OC4v4 -- for that, and for LS2, use
    :func:`oc4v4`.  The nearest available band is used for each wavelength.

    Parameters
    ----------
    wave : numpy.ndarray
        Wavelengths [nm], shape ``(L,)``.
    Rrs : numpy.ndarray
        Remote-sensing reflectance [sr^-1], shape ``(L,)``.

    Returns
    -------
    float
        Chlorophyll-a concentration [mg m^-3].
    """

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


def oc4v4(wave, Rrs, max_offset:float=10.):
    """OC4 version 4 (O'Reilly et al. 2000), vectorized over spectra.

    .. math::

       \\log_{10} \\mathrm{Chl} = 0.366 - 3.067 R + 1.930 R^2
                                 + 0.649 R^3 - 1.532 R^4,

    with :math:`R = \\log_{10}[\\max(R_{rs}(443), R_{rs}(490), R_{rs}(510))
    / R_{rs}(555)]`.  A pure quartic: there is no additive constant, unlike
    the 1998 form in :func:`oc4`.  This is the chlorophyll algorithm cited by
    LS2 (Loisel et al. 2018).

    The nearest available band is used for each of the four SeaWiFS
    wavelengths, as in :func:`oc4`, but a band more than ``max_offset`` nm
    away raises rather than being used silently.

    Parameters
    ----------
    wave : array_like
        Wavelengths [nm], shape ``(L,)``.
    Rrs : array_like
        Remote-sensing reflectance [sr^-1], shape ``(..., L)``.
    max_offset : float, optional
        Largest tolerated distance [nm] between a requested band and the
        nearest one in ``wave``.  Default 10.

    Returns
    -------
    numpy.ndarray
        Chlorophyll-a concentration [mg m^-3], shape ``Rrs.shape[:-1]``.
        NaN where the band ratio is not positive and finite.

    Raises
    ------
    ValueError
        If any of 443, 490, 510 or 555 nm has no band within ``max_offset``.

    References
    ----------
    O'Reilly, J. E. et al. (2000), Ocean color chlorophyll a algorithms for
    SeaWiFS, OC2, and OC4: Version 4, in *SeaWiFS Postlaunch Calibration and
    Validation Analyses, Part 3*, NASA Tech. Memo. 2000-206892, Vol. 11,
    edited by S. B. Hooker and E. R. Firestone, pp. 9-23 (Eq. 4).
    """
    wave = np.asarray(wave, dtype=float).ravel()
    Rrs = np.asarray(Rrs, dtype=float)

    idx = []
    for band in OC4_BANDS:
        i = int(np.argmin(np.abs(wave - band)))
        if abs(wave[i] - band) > max_offset:
            raise ValueError(f'No band within {max_offset} nm of {band:g} nm '
                             f'(nearest is {wave[i]:g} nm)')
        idx.append(i)
    i443, i490, i510, i555 = idx

    blue = np.maximum(np.maximum(Rrs[..., i443], Rrs[..., i490]),
                      Rrs[..., i510])
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = blue / Rrs[..., i555]
        good = np.isfinite(ratio) & (ratio > 0.)
        R = np.log10(np.where(good, ratio, np.nan))

    a0, a1, a2, a3, a4 = OC4V4_COEFFS
    return 10.**(a0 + a1*R + a2*R**2 + a3*R**3 + a4*R**4)
