"""Particulate scattering coefficients from chlorophyll.

LS2 (Loisel et al. 2018) needs the particulate scattering coefficient ``b_p``
only to form ``eta = b_w / (b_p + b_w)``, so it is a second-order input.  When
``b_p`` is not measured the authors estimate it from chlorophyll, and
:func:`bp_from_chla` reproduces their estimate exactly.
"""

from __future__ import annotations

import numpy as np

#: Amplitude of ``b_p(660)`` [m^-1 (mg m^-3)^-EXPONENT]; Loisel & Morel
#: (1998) Eq. 6, the homogeneous-layer regression of their Table 2.
BP660_AMPLITUDE = 0.347

#: Exponent of chlorophyll in ``b_p(660)``; Loisel & Morel (1998) Eq. 6.
BP660_EXPONENT = 0.766

#: Reference wavelength [nm] of the amplitude.
BP_WAVE0 = 660.


def bp_from_chla(wave, Chl):
    """Particulate scattering coefficient from chlorophyll, as LS2's authors do.

    Implements ``bp_from_Chla.m`` from the authors' ``LS2_Distribution``
    exactly:

    .. math::

       b_p(660) = 0.347\\,[\\mathrm{Chl}]^{0.766}, \\qquad
       b_p(\\lambda) = b_p(660)\\,(\\lambda / 660)^{-1}.

    **The spectral shape is a plain** :math:`\\lambda^{-1}` **as implemented by
    the authors**, not the chlorophyll-dependent exponent of Morel & Maritorena
    (2001), even though ``bp_from_Chla.m`` names MM01 as its source.  This is
    why the ``Input bp`` column of the authors' reference vector is an exact
    :math:`\\lambda^{-1}` power law (to 2e-15).

    The amplitude and exponent are Eq. 6 of Loisel & Morel (1998): their
    regression for the homogeneous (mixed) layer, :math:`r^2 = 0.88`,
    :math:`N = 850` (Table 2).  Strictly, that regression is fitted to the
    particle *attenuation* coefficient :math:`c_p(660)`.  The paper argues
    that :math:`c_p` "can be safely considered as equivalent to
    :math:`b_p`" at 660 nm, where particulate absorption is small, and that
    is the identification the authors make.  Loisel & Morel themselves use a
    :math:`\\lambda^{-1}` dependence to carry scattering from 550 to 660 nm,
    so the shape has a precedent in the same paper, though not as part of
    Eq. 6.

    Parameters
    ----------
    wave : array_like
        Light wavelengths [nm], shape ``(L,)`` or scalar.
    Chl : array_like
        Chlorophyll-a concentration [mg m^-3], shape ``(N,)`` or scalar.

    Returns
    -------
    numpy.ndarray
        ``b_p`` [m^-1].  Shape ``(N, L)`` for array ``Chl``, ``(L,)`` for
        scalar ``Chl``, and a 0-d array when both are scalar.  Negative or
        non-finite ``Chl`` gives NaN.

    References
    ----------
    Loisel, H. and A. Morel (1998), Light scattering and chlorophyll
    concentration in case 1 waters: A reexamination, *Limnol. Oceanogr.*,
    43, 847-858.

    Morel, A. and S. Maritorena (2001), Bio-optical properties of oceanic
    waters: A reappraisal, *J. Geophys. Res.*, 106, 7163-7180.

    SIO Ocean Optics Research Laboratory, ``LS2_Distribution``,
    ``bp_from_Chla.m`` (A. Taylor, S. Hart and M. Kehrli, 2022; MIT
    licence), https://github.com/SIO-Ocean-Optics-Research-Laboratory/LS2_Distribution
    """
    wave = np.asarray(wave, dtype=float)
    Chl = np.asarray(Chl, dtype=float)

    valid = np.isfinite(Chl) & (Chl >= 0.)
    bp660 = BP660_AMPLITUDE * np.where(valid, Chl, np.nan) ** BP660_EXPONENT

    shape = (wave / BP_WAVE0) ** -1
    if bp660.ndim == 0:
        return bp660 * shape
    return bp660[..., None] * shape
