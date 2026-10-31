"""The LS2 inversion model of Loisel et al. (2018).

LS2 turns a remote-sensing reflectance ``Rrs`` and an average diffuse
attenuation coefficient ``<Kd>_1`` into the total absorption ``a`` and total
backscattering ``bb`` coefficients, using look-up tables of polynomial
coefficients indexed by ``eta`` (the pure-water fraction of total scattering)
and ``muw`` (the cosine of the refracted solar beam).  It is a *direct*
algorithm: there is no fit, no likelihood and no misfit.

Two entry points are provided.

``ls2_invert``
    The vectorized workhorse.  It runs over an ``(N, L)`` block of samples and
    wavelengths, iterates the Raman correction to convergence, and reports
    per-cell diagnostic flags instead of raising warnings.  This is what
    callers should use.

``LS2_main``
    A thin scalar wrapper preserving the historical 5-tuple contract
    ``(a, anw, bb, bbp, kappa)``.  It defaults to ``max_iter=1``, which is what
    the authors' MATLAB distribution does, so it remains a faithful oracle.

Differences from the authors' MATLAB reference, all deliberate
----------------------------------------------------------------

* **All four ``a`` corners are recomputed in the Raman branch.**  The earlier
  Python port recomputed only ``a00`` and reused three stale corners; the
  MATLAB ``LS2_main.m`` recomputes all four.  This was a porting defect.
* **Iteration.**  The paper (Table 1, step 9) says to iterate the Raman
  correction to convergence; the authors' code performs a single pass.
  ``ls2_invert`` iterates by default and ``LS2_main`` does not, so each mode
  reproduces one of the two references exactly.  Following Loisel & Stramski
  (2000) Eq. 22, ``kappa`` is applied to the *original* ``Rrs`` on every pass —
  the correction is not cumulative.
* **kappa outside the table's wavelength range is NaN**, not the clamped
  end-of-table value that ``np.interp`` would return.  The MATLAB
  ``interp1(..., 'linear')`` also returns NaN there.  The table stops at
  702 nm, so every wavelength above it is unavailable rather than wrong.
* **Off-grid returns NaN**, not ``None``.  The earlier port's bare ``return``
  handed back ``None`` and broke its caller's 5-tuple unpacking.
* **No per-cell warnings.**  Negative solutions, off-grid ``eta``/``muw`` and
  out-of-range ``kappa`` are counted in boolean flag arrays.  On an 800k-cell
  corpus the warnings were a storm that hid everything else.

References
----------
Loisel, H., D. Stramski, D. Dessailly, C. Jamet, L. Li and R. A. Reynolds
(2018), An inverse model for estimating the optical absorption and
backscattering coefficients of seawater from remote-sensing reflectance over a
broad range of oceanic and coastal marine environments, *J. Geophys. Res.
Oceans*, 123, 2141-2171, doi:10.1002/2017JC013632.

Loisel, H. and D. Stramski (2000), Estimation of the inherent optical
properties of natural waters from the irradiance attenuation coefficient and
reflectance in the presence of Raman scattering, *Appl. Opt.*, 39, 3001-3011.

Version History
---------------
2018-04-04: Original implementation in C, D. Dessailly.
2020-03-23: Original MATLAB version, D. Jorge.
2022-09-01: Revised MATLAB version, M. Kehrli.
2022-11-03: Final revised MATLAB version, M. Kehrli, R. A. Reynolds and
D. Stramski.
2023-06-22: Converted to Python by JXP and Claude+.
2026-09-23: Corrected, iterated and vectorized (``ls2_invert``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Refractive index of seawater used to refract the solar beam (Table 1, step 1).
N_WATER = 1.34

#: Absolute tolerance for snapping an input onto a look-up table edge.
#:
#: The stored ``muw`` nodes are the nominal solar zenith angles 0, 10, ... 70
#: degrees rounded to six decimal places, so ``sza = 70`` evaluates to a ``muw``
#: that sits 4.9e-7 *below* the table's last node and would otherwise be
#: rejected as off-grid.  1e-5 clears that rounding by more than an order of
#: magnitude while remaining 500x smaller than the narrowest cell.
_EDGE_ATOL = 1.0e-5


@dataclass
class LS2Result:
    """Outputs of :func:`ls2_invert`, every array of the broadcast shape.

    The four flags are how a caller learns why a cell came back NaN.  The
    vectorized path emits no warnings -- one per cell is 800k warnings on the
    benchmark corpus -- so these arrays, and :meth:`counts`, carry the
    diagnosis instead.
    """

    #: Total absorption coefficient [m^-1].
    a: np.ndarray
    #: Non-water absorption coefficient, ``a - aw`` [m^-1].
    anw: np.ndarray
    #: Total backscattering coefficient [m^-1].
    bb: np.ndarray
    #: Particulate backscattering coefficient, ``bb - bw/2`` [m^-1].
    bbp: np.ndarray
    #: Raman correction factor applied to ``Rrs`` [dim], always the one that
    #: produced the returned ``a`` and ``bb``.  1.0 where the correction was
    #: not requested, NaN where it could not be evaluated even once.
    kappa: np.ndarray
    #: Number of completed Raman passes.  0 when ``raman=False``.
    n_iter: np.ndarray
    #: ``eta`` or ``muw`` fell outside the look-up table; all outputs NaN.
    off_grid: np.ndarray
    #: ``kappa`` could not be evaluated, either because ``bb/a`` left the
    #: table's admissible range at that wavelength or because the wavelength
    #: itself is outside the table's 302-702 nm span.  The iteration stops
    #: there and the coefficients of the last completed pass are kept.
    #: ``kappa`` is then the value that produced them -- the last *applied*
    #: one -- or NaN if the failure came on the first pass and no correction
    #: was ever applied.
    kappa_out_of_range: np.ndarray
    #: At least one of ``a``, ``anw``, ``bb``, ``bbp`` came out negative.
    #: Whether the offending value was replaced by NaN is governed by
    #: ``clip_negative``.
    negative: np.ndarray
    #: The Raman iteration ended without ``|d(bb/a)|/(bb/a) < tol``, either
    #: because ``max_iter`` was reached or because ``kappa`` left its range.
    #: Always False when ``raman=False``.
    not_converged: np.ndarray

    @property
    def converged(self) -> np.ndarray:
        """Complement of :attr:`not_converged`, for callers that prefer it."""
        return ~self.not_converged

    def counts(self) -> dict:
        """Return the number of cells raised by each flag.

        The vectorized path deliberately emits no warnings, so this is how a
        caller learns that, say, 19% of its cells lost their Raman correction.
        """
        return {
            'n_cells': int(self.a.size),
            'off_grid': int(self.off_grid.sum()),
            'kappa_out_of_range': int(self.kappa_out_of_range.sum()),
            'negative': int(self.negative.sum()),
            'not_converged': int(self.not_converged.sum()),
        }


def _bracket(param: np.ndarray, nodes: np.ndarray):
    """Leftmost bracketing index of ``param`` in a strictly monotonic grid.

    Vectorized replacement for the scan in :func:`LS2_seek_pos`.  The final
    cell is treated as right-closed, so a value landing exactly on the last
    node is bracketed by the last cell rather than falling through the loop
    (which is the ``UnboundLocalError`` the scalar version used to raise at
    ``eta = 0.2`` and at the ``muw`` of ``sza = 70`` degrees).

    Parameters
    ----------
    param : numpy.ndarray
        Values to locate.
    nodes : numpy.ndarray
        Strictly monotonic grid, ascending (``eta``) or descending (``muw``).

    Returns
    -------
    idx : numpy.ndarray of int
        Leftmost index of the bracketing cell, in ``[0, len(nodes) - 2]``.
        Meaningless where ``off`` is True.
    frac : numpy.ndarray
        Position within the cell, clipped to ``[0, 1]``.
    off : numpy.ndarray of bool
        Values outside the grid (or not finite).
    """
    param = np.asarray(param, dtype=float)
    nodes = np.asarray(nodes, dtype=float).ravel()

    # Work in an ascending frame so one searchsorted serves both grids.
    ascending = nodes[-1] > nodes[0]
    key = nodes if ascending else -nodes
    x = param if ascending else -param

    lo, hi = key[0], key[-1]
    # Snap values that miss an edge only by the tables' rounding.
    x = np.where((x < lo) & (x >= lo - _EDGE_ATOL), lo, x)
    x = np.where((x > hi) & (x <= hi + _EDGE_ATOL), hi, x)

    off = ~np.isfinite(x) | (x < lo) | (x > hi)

    idx = np.searchsorted(key, np.where(off, lo, x), side='right') - 1
    idx = np.clip(idx, 0, key.size - 2)

    left = key[idx]
    frac = np.clip((np.where(off, lo, x) - left) / (key[idx + 1] - left), 0., 1.)
    return idx, frac, off


def _bilinear(v00, v01, v10, v11, fe, fm):
    """Bilinear blend of the four bracketing corners.

    ``v<i><j>`` is the value at ``eta`` offset ``i`` and ``muw`` offset ``j``,
    matching the naming in the authors' MATLAB.  ``fe`` and ``fm`` are the
    fractional positions within the ``eta`` and ``muw`` cells.

    The corners must be the *derived* ``a`` or ``bb``, never the polynomial
    coefficients: ``Kd / P(Rrs)`` interpolated is not the interpolation of
    ``P`` evaluated at ``Rrs``, and swapping the two silently breaks agreement
    with the reference implementation.
    """
    return ((1. - fe) * ((1. - fm) * v00 + fm * v01)
            + fe * ((1. - fm) * v10 + fm * v11))


def _solve(Rrs, Kd, A, B, ie, im, fe, fm):
    """Evaluate Eqs. 9 and 8 at the four corners and blend them.

    Parameters
    ----------
    Rrs, Kd : numpy.ndarray
        Reflectance [sr^-1] and average attenuation [m^-1], broadcast shape.
    A, B : numpy.ndarray
        The ``(21, 8, 4)`` absorption and ``(21, 8, 3)`` backscattering
        coefficient tables.
    ie, im : numpy.ndarray of int
        Bracketing ``eta`` and ``muw`` indices.
    fe, fm : numpy.ndarray
        Fractional positions within those cells.

    Returns
    -------
    a, bb : numpy.ndarray
        Total absorption and backscattering coefficients [m^-1].
    """
    r2 = Rrs * Rrs
    r3 = r2 * Rrs

    def _a(i, j):
        # Eq. 9: a = Kd / (c0 + c1 Rrs + c2 Rrs^2 + c3 Rrs^3)
        c = A[i, j]
        return Kd / (c[..., 0] + c[..., 1] * Rrs + c[..., 2] * r2
                     + c[..., 3] * r3)

    def _bb(i, j):
        # Eq. 8: bb = Kd (c0 Rrs + c1 Rrs^2 + c2 Rrs^3)
        c = B[i, j]
        return Kd * (c[..., 0] * Rrs + c[..., 1] * r2 + c[..., 2] * r3)

    a = _bilinear(_a(ie, im), _a(ie, im + 1),
                  _a(ie + 1, im), _a(ie + 1, im + 1), fe, fm)
    bb = _bilinear(_bb(ie, im), _bb(ie, im + 1),
                   _bb(ie + 1, im), _bb(ie + 1, im + 1), fe, fm)
    return a, bb


def _kappa_terms(wave, rLUT):
    """Interpolate the kappa cubic's coefficients and bounds to ``wave``.

    ``LS2_calc_kappa`` evaluates the cubic at all 101 table wavelengths and
    then interpolates the *result* to the input wavelength.  Linear
    interpolation is linear in the tabulated values and the powers of ``bb/a``
    are common to every row, so interpolating the four coefficients first and
    evaluating the cubic once is algebraically identical -- and costs
    ``O(L)`` instead of ``O(N x L x 101)``, which is what makes an 800k-cell
    corpus tractable.

    Parameters
    ----------
    wave : numpy.ndarray
        Wavelengths [nm], broadcast shape.
    rLUT : numpy.ndarray
        The ``(101, 7)`` Raman table: wavelength, three cubic coefficients, the
        constant term, then the minimum and maximum admissible ``bb/a``.

    Returns
    -------
    coef : list of numpy.ndarray
        The four cubic coefficients at ``wave``, highest power first.
    mins, maxs : numpy.ndarray
        Admissible ``bb/a`` range at ``wave``.
    in_table : numpy.ndarray of bool
        Whether ``wave`` lies inside the table's span (302-702 nm).  Outside
        it ``kappa`` is NaN rather than the clamped edge value.
    """
    lam = rLUT[:, 0]
    coef = [np.interp(wave, lam, rLUT[:, k]) for k in (1, 2, 3, 4)]
    mins = np.interp(wave, lam, rLUT[:, 5])
    maxs = np.interp(wave, lam, rLUT[:, 6])
    in_table = (wave >= lam[0]) & (wave <= lam[-1])
    return coef, mins, maxs, in_table


def ls2_invert(Rrs, Kd, aw, bw, bp, sza, wave, LS2_LUT, *, raman=True,
               tol=1.0e-3, max_iter=10, clip_negative=False,
               muw=None) -> LS2Result:
    """Run LS2 over a block of samples and wavelengths.

    Parameters
    ----------
    Rrs : array_like
        Spectral remote-sensing reflectance [sr^-1], shape ``(N, L)`` (a bare
        ``(L,)`` spectrum is accepted and treated as one sample).
    Kd : array_like
        Average diffuse attenuation coefficient of downwelling planar
        irradiance between the surface and the first attenuation depth
        [m^-1], shape ``(N, L)``.
    aw, bw : array_like
        Pure seawater absorption and scattering coefficients [m^-1], shape
        ``(L,)``.
    bp : array_like
        Particulate scattering coefficient [m^-1], shape ``(N, L)``.
    sza : array_like
        Solar zenith angle [deg], shape ``(N,)`` or scalar.
    wave : array_like
        Light wavelengths [nm], shape ``(L,)``.
    LS2_LUT : mapping
        The look-up tables, as returned by :func:`ocpy.ls2.io.load_LUT`:
        ``eta`` ``(21,)``, ``muw`` ``(8,)``, ``a`` ``(21, 8, 4)``,
        ``bb`` ``(21, 8, 3)`` and ``kappa`` ``(101, 7)``.
    raman : bool, optional
        Apply the Raman scattering correction.  Default True.
    tol : float, optional
        Relative convergence tolerance on ``bb/a``.  Default 1e-3, the
        criterion of Loisel & Stramski (2000).
    max_iter : int, optional
        Cap on Raman passes.  Default 10.  Set to 1 to reproduce the authors'
        single-pass MATLAB distribution exactly.
    clip_negative : bool, optional
        Replace negative outputs by NaN, each coefficient independently, as
        the authors do.  Default False: the raw values are returned and the
        ``negative`` flag records where they occurred, because a benchmark
        needs to see the sign of the failure rather than only its absence.
    muw : array_like, optional
        Enter the look-up tables at this ``muw`` instead of the refracted
        cosine of ``sza`` (Table 1, step 1), broadcastable to ``(N, L)`` --
        so it may differ per wavelength.  For diagnostics only: an
        *effective* cosine taken from a radiative-transfer light field
        tests whether a bias is illumination bookkeeping rather than
        coefficient error (IOPtics ls2 Q9).  ``sza`` is then ignored.
        Values outside the table's ``muw`` span are off-grid.

    Returns
    -------
    LS2Result
        Coefficients, ``kappa``, the pass count and four per-cell flags.

    Notes
    -----
    ``kappa`` is applied to the *original* ``Rrs`` on each pass, not to the
    previously corrected value; the correction is a property of the water, not
    a cumulative rescaling (Loisel & Stramski 2000, Eq. 22).

    If ``kappa`` leaves its admissible range on a later pass, the iteration
    stops there, the coefficients of the last completed pass are kept, and
    ``kappa`` is reported as the value last *applied* -- so the returned
    ``kappa`` always reproduces the returned ``a`` and ``bb`` -- with
    ``kappa_out_of_range`` and ``not_converged`` set.  On the first pass this
    reduces exactly to the authors' behaviour: no correction at all,
    uncorrected coefficients, ``kappa`` NaN.
    """
    Rrs = np.atleast_2d(np.asarray(Rrs, dtype=float))
    Kd = np.atleast_2d(np.asarray(Kd, dtype=float))
    bp = np.atleast_2d(np.asarray(bp, dtype=float))
    aw = np.asarray(aw, dtype=float).reshape(1, -1)
    bw = np.asarray(bw, dtype=float).reshape(1, -1)
    wave = np.asarray(wave, dtype=float).reshape(1, -1)
    sza = np.asarray(sza, dtype=float).reshape(-1, 1)

    shape = np.broadcast_shapes(Rrs.shape, Kd.shape, bp.shape, aw.shape,
                                bw.shape, wave.shape, sza.shape)
    Rrs, Kd, bp, aw, bw, wave, sza = (
        np.broadcast_to(x, shape).astype(float, copy=True)
        for x in (Rrs, Kd, bp, aw, bw, wave, sza))

    # Pull the tables out of the (possibly lazy) npz exactly once.
    eta_nodes = np.asarray(LS2_LUT['eta'], dtype=float).ravel()
    muw_nodes = np.asarray(LS2_LUT['muw'], dtype=float).ravel()
    A = np.asarray(LS2_LUT['a'], dtype=float)
    B = np.asarray(LS2_LUT['bb'], dtype=float)
    rLUT = np.asarray(LS2_LUT['kappa'], dtype=float)

    if eta_nodes.size != 21 or not np.all(np.diff(eta_nodes) > 0):
        raise ValueError('eta look-up table must be 21 values in ascending order')
    if muw_nodes.size != 8 or not np.all(np.diff(muw_nodes) < 0):
        raise ValueError('muw look-up table must be 8 values in descending order')

    # Step 1: muw, the cosine of the refracted solar beam -- unless the
    # caller supplies an effective one (a diagnostic; see ``muw`` above).
    if muw is None:
        muw = np.cos(np.arcsin(np.sin(np.deg2rad(sza)) / N_WATER))
    else:
        muw = np.broadcast_to(np.asarray(muw, dtype=float), shape)

    # Steps 3 & 4: total scattering and its pure-water fraction.
    with np.errstate(divide='ignore', invalid='ignore'):
        eta = bw / (bp + bw)

    ie, fe, off_eta = _bracket(eta, eta_nodes)
    im, fm, off_muw = _bracket(muw, muw_nodes)
    off_grid = off_eta | off_muw

    # Steps 5 & 7: a and bb from Eqs. 9 and 8, uncorrected.
    with np.errstate(divide='ignore', invalid='ignore'):
        a, bb = _solve(Rrs, Kd, A, B, ie, im, fe, fm)
    a = np.where(off_grid, np.nan, a)
    bb = np.where(off_grid, np.nan, bb)

    kappa = np.ones(shape)
    n_iter = np.zeros(shape, dtype=int)
    kappa_oor = np.zeros(shape, dtype=bool)
    converged = np.ones(shape, dtype=bool)

    # Step 9: the Raman scattering correction, iterated.
    if raman:
        converged = np.zeros(shape, dtype=bool)
        coef, mins, maxs, in_table = _kappa_terms(wave, rLUT)
        active = np.isfinite(a) & np.isfinite(bb) & (a != 0.)
        converged |= ~active  # nothing to iterate on; do not flag as stalled

        for _ in range(max(int(max_iter), 0)):
            if not active.any():
                break
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = bb / a

            k = (coef[0] * ratio ** 3 + coef[1] * ratio ** 2
                 + coef[2] * ratio + coef[3])
            usable = in_table & (ratio >= mins) & (ratio <= maxs) & np.isfinite(k)

            stalled = active & ~usable
            kappa_oor |= stalled
            # Keep the last applied kappa; NaN only if none was ever applied.
            kappa = np.where(stalled & (n_iter == 0), np.nan, kappa)
            active = active & usable
            if not active.any():
                break

            kappa = np.where(active, k, kappa)
            with np.errstate(divide='ignore', invalid='ignore'):
                a_new, bb_new = _solve(Rrs * kappa, Kd, A, B, ie, im, fe, fm)
            a = np.where(active, a_new, a)
            bb = np.where(active, bb_new, bb)
            n_iter += active

            with np.errstate(divide='ignore', invalid='ignore'):
                moved = np.abs(bb / a - ratio) / np.abs(bb / a)
            done = active & (moved < tol)
            converged |= done
            active = active & ~done

    # Steps 6 & 8: the non-water and particulate parts.
    anw = a - aw
    bbp = bb - bw / 2.

    negative = ((a < 0.) | (anw < 0.) | (bb < 0.) | (bbp < 0.))
    if clip_negative:
        a = np.where(a < 0., np.nan, a)
        anw = np.where(anw < 0., np.nan, anw)
        bb = np.where(bb < 0., np.nan, bb)
        bbp = np.where(bbp < 0., np.nan, bbp)

    kappa = np.where(off_grid, np.nan, kappa)

    return LS2Result(a=a, anw=anw, bb=bb, bbp=bbp, kappa=kappa,
                     n_iter=n_iter, off_grid=off_grid,
                     kappa_out_of_range=kappa_oor, negative=negative,
                     not_converged=~converged)


def LS2_main(sza: float, lambda_: float, Rrs: float, Kd: float, aw: float,
             bw: float, bp: float, LS2_LUT, Flag_Raman, *, tol: float = 1.0e-3,
             max_iter: int = 1, clip_negative: bool = True):
    """Scalar LS2 inversion; a thin wrapper over :func:`ls2_invert`.

    Preserved for the historical 5-tuple contract and as the oracle against the
    authors' MATLAB distribution, which is single-pass.  ``max_iter`` therefore
    defaults to 1 here and to 10 in :func:`ls2_invert`.  New code should call
    :func:`ls2_invert`, which is faster by three orders of magnitude on a large
    corpus and reports why a cell came back NaN.

    Parameters
    ----------
    sza : float
        Solar zenith angle [deg].
    lambda_ : float
        Input light wavelength [nm].
    Rrs : float
        Spectral remote-sensing reflectance [sr^-1] at ``lambda_``.
    Kd : float
        Average spectral attenuation coefficient of downwelling planar
        irradiance [m^-1] between the surface and the first attenuation depth.
    aw, bw : float
        Pure seawater absorption and scattering coefficients [m^-1].
    bp : float
        Particulate scattering coefficient [m^-1].
    LS2_LUT : mapping
        Look-up tables, as returned by :func:`ocpy.ls2.io.load_LUT`.
    Flag_Raman : bool
        Apply the Raman scattering correction to ``Rrs``.
    tol : float, optional
        Relative convergence tolerance on ``bb/a``; irrelevant at
        ``max_iter=1``.
    max_iter : int, optional
        Cap on Raman passes.  Default 1, the authors' behaviour.
    clip_negative : bool, optional
        Replace negative outputs by NaN.  Default True, the authors' behaviour.

    Returns
    -------
    a, anw, bb, bbp, kappa : float
        Total absorption, non-water absorption, total backscattering and
        particulate backscattering coefficients [m^-1], and the Raman
        correction factor [dim].  All NaN if ``eta`` or ``muw`` is off-grid.
    """
    res = ls2_invert(np.array([[Rrs]]), np.array([[Kd]]), np.array([aw]),
                     np.array([bw]), np.array([[bp]]), np.array([sza]),
                     np.array([lambda_]), LS2_LUT, raman=bool(Flag_Raman),
                     tol=tol, max_iter=max_iter, clip_negative=clip_negative)
    return (res.a[0, 0], res.anw[0, 0], res.bb[0, 0], res.bbp[0, 0],
            res.kappa[0, 0])


def LS2_seek_pos(param: float, LUT: np.ndarray, itype: str):
    """Find the leftmost bracketing position of ``param`` in its look-up table.

    Scalar remake of the LS2 ``seek_pos`` subroutine, kept for the published
    API.  :func:`ls2_invert` uses the vectorized :func:`_bracket` instead.

    Two behaviours differ from the MATLAB original, both bug fixes.  A value
    landing exactly on the last node now returns the last cell rather than
    leaving ``idx`` unassigned (MATLAB errors; the previous Python raised
    ``UnboundLocalError``), and a value that misses an edge only by the tables'
    six-decimal rounding is snapped onto it -- without which ``sza = 70``
    degrees, a nominal grid node, falls 4.9e-7 outside the ``muw`` table.

    Parameters
    ----------
    param : float
        Input ``muw`` or ``eta`` value.
    LUT : numpy.ndarray
        Look-up table of ``muw`` values (8, descending) or ``eta`` values
        (21, ascending).
    itype : str
        ``'muw'`` or ``'eta'``.

    Returns
    -------
    int or float
        Leftmost index of the bracketing cell, or NaN if ``param`` is outside
        the table.
    """
    LUT = np.asarray(LUT, dtype=float).ravel()
    if itype == 'muw':
        if LUT.size != 8 or not np.all(np.diff(LUT) < 0):
            raise ValueError('Look-up table for mu_w must be a 8x1 array '
                             'sorted in descending order')
    elif itype == 'eta':
        if LUT.size != 21 or not np.all(np.diff(LUT) > 0):
            raise ValueError('Look-up table for eta must be a 21x1 array '
                             'sorted in ascending order')
    else:
        raise ValueError("itype must be 'muw' or 'eta'")

    idx, _, off = _bracket(np.asarray(param, dtype=float), LUT)
    if bool(off):
        return np.nan
    return int(idx)


def LS2_calc_kappa(bb_a: float, lam: float, rLUT: np.ndarray):
    """Interpolate the Raman correction factor ``kappa`` from its look-up table.

    Scalar remake of the LS2 ``calc_kappa`` subroutine, kept for the published
    API.  :func:`ls2_invert` uses :func:`_kappa_terms` instead.

    Unlike the previous Python port this returns NaN above the table's
    702 nm limit rather than silently clamping to the last row, which is what
    the MATLAB ``interp1(..., 'linear')`` does and what the 750 nm end of a
    hyperspectral corpus needs it to do.

    Parameters
    ----------
    bb_a : float
        Backscattering to absorption ratio from the LS2 inversion.
    lam : float
        Wavelength [nm] of that ratio.
    rLUT : numpy.ndarray
        The ``(101, 7)`` Raman look-up table.

    Returns
    -------
    float
        ``kappa`` [dim], or NaN if ``bb_a`` is outside the admissible range at
        ``lam`` or if ``lam`` is outside the table.
    """
    rLUT = np.asarray(rLUT, dtype=float)
    coef, mins, maxs, in_table = _kappa_terms(np.asarray(lam, dtype=float),
                                              rLUT)
    if not bool(in_table) or not (mins <= bb_a <= maxs):
        return np.nan
    return float(coef[0] * bb_a ** 3 + coef[1] * bb_a ** 2
                 + coef[2] * bb_a + coef[3])
