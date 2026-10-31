"""Re-derive LS2's ``a`` and ``bb`` coefficient tables from a radiative-transfer corpus.

LS2 (Loisel et al. 2018) relates ``Rrs`` and ``<Kd>_1`` to the total
absorption and backscattering through two polynomials whose coefficients are
tabulated on ``eta`` (21 nodes, ``eta = b_w / b``) and ``mu_w`` (8 nodes,
theta_s = 0-70 degrees)::

    <Kd>_1 / a  = c0 + c1 Rrs + c2 Rrs^2 + c3 Rrs^3          (Eq. 9)
    bb / <Kd>_1 =      d0 Rrs + d1 Rrs^2 + d2 Rrs^3          (Eq. 8)

This module refits those coefficients from a corpus where ``a``, ``bb``,
``eta``, ``Rrs`` and ``<Kd>_1`` are all known (IOPtics ls2 task 12 uses L23's
elastic realization).  It does not tabulate node by node.  A corpus such as
L23 has only three solar zeniths (it lands on 3 of the 8 ``mu_w`` nodes) and
an ``eta`` distribution that is 50:1 unbalanced across the table, so every
coefficient is modelled as a **smooth low-order function** of both:

    c_k(eta, mu_w) = sum_i sum_j  beta_kij  s^i  g_j(mu_w),   s = sqrt(eta / 0.2)

with ``g`` one of the ``mu_w`` bases of :data:`MUW_BASES`, chosen per table.
The physics suggests ``(1, 1/mu_w)`` for ``<Kd>_1/a``, whose ``Rrs -> 0`` limit
is ``1/mu_w`` (the slant path), and ``(1, mu_w)`` for ``bb/<Kd>_1``, which
scales as ``mu_w`` for the same reason.  On L23 a held-out-zenith check
confirms both (IOPtics ls2 task 12).  ``sqrt`` spreads the
crowded low-``eta`` end of the axis, where most of the data lie.  Both
targets are linear in ``beta``, so the fit is one weighted linear
least-squares solve per table, minimising the *relative* error of
``<Kd>_1/a`` and ``bb/<Kd>_1`` (and so, to first order, of ``a`` and ``bb``).
That is deterministic, with no seed and no optimiser.

:meth:`SmoothCoefficients.to_lut` evaluates the model on the published
21 x 8 node grid, giving a drop-in replacement for the ``a`` and ``bb``
arrays of ``LS2_LUT.npz``, so :func:`ocpy.ls2.ls2_main.ls2_invert` uses it
unchanged.  The bilinear interpolation between nodes of a smooth function
costs a measurable but small error, which IOPtics reports.  Nodes beyond the
corpus (theta_s = 70 degrees for L23) are extrapolated by the ``mu_w``
basis.  That is recorded in the table's provenance and is the reason the
basis is chosen by a held-out-zenith check rather than by fit quality.

The published cubic's form is kept on purpose: on L23 a cubic in ``Rrs`` fits
``<Kd>_1/a`` with R^2 ~ 0.99 (IOPtics ls2 Q11), so a new functional form is not
called for.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: ``eta`` is scaled by this before the square root (the published table's top).
ETA_SCALE = 0.2

#: Bases in ``mu_w`` for the coefficients' geometry dependence.
#:
#: ``'inv'`` = ``(1, 1/mu_w)``: the form of the limiting relation ``c0 -> 1/mu_w``.
#: ``'lin'`` = ``(1, mu_w)``.  ``'const'`` = ``(1,)``: no geometry dependence.
MUW_BASES = {
    'inv': lambda m: np.stack([np.ones_like(m), 1.0 / m], axis=-1),
    'lin': lambda m: np.stack([np.ones_like(m), m], axis=-1),
    'const': lambda m: np.ones_like(m)[..., None],
}


def eta_basis(eta, degree):
    """``(…, degree+1)`` powers of ``s = sqrt(eta / ETA_SCALE)``."""
    s = np.sqrt(np.clip(np.asarray(eta, dtype=float), 0.0, None) / ETA_SCALE)
    return np.stack([s ** i for i in range(degree + 1)], axis=-1)


def _design(rrs, eta, muw, powers, eta_degree, muw_basis):
    """Design matrix: one column per (Rrs power, eta term, mu_w term)."""
    E = eta_basis(eta, eta_degree)                      # (n, ne)
    M = MUW_BASES[muw_basis](np.asarray(muw, float))    # (n, nm)
    cols = [(rrs ** k)[:, None, None] * E[:, :, None] * M[:, None, :]
            for k in powers]
    return np.concatenate([c.reshape(len(rrs), -1) for c in cols], axis=1)


@dataclass
class SmoothCoefficients:
    """A fitted pair of coefficient models (``a`` and ``bb``).

    Attributes
    ----------
    beta_a : numpy.ndarray
        ``(4, eta_degree+1, n_muw_terms)`` for ``c0..c3``.
    beta_b : numpy.ndarray
        ``(3, eta_degree+1, n_muw_terms)`` for ``d0..d2``.
    eta_degree : int
    muw_basis : tuple of str
        ``(basis for a, basis for bb)``, keys of :data:`MUW_BASES`.
    domain : dict
        Ranges of the training corpus (``eta``, ``mu_w``, ``Rrs``, ``b/a``);
        outside them the model extrapolates.
    meta : dict
        Provenance.
    """

    beta_a: np.ndarray
    beta_b: np.ndarray
    eta_degree: int
    muw_basis: tuple
    domain: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def coefficients(self, eta, muw):
        """``(c (…, 4), d (…, 3))`` at ``(eta, mu_w)``."""
        E = eta_basis(eta, self.eta_degree)
        m = np.asarray(muw, float)
        c = np.einsum('kij,...i,...j->...k', self.beta_a, E,
                      MUW_BASES[self.muw_basis[0]](m))
        d = np.einsum('kij,...i,...j->...k', self.beta_b, E,
                      MUW_BASES[self.muw_basis[1]](m))
        return c, d

    def invert(self, rrs, kd, eta, muw):
        """``(a, bb)`` from the smooth model directly (no table, no Raman)."""
        c, d = self.coefficients(eta, muw)
        r = np.asarray(rrs, float)
        pa = c[..., 0] + c[..., 1] * r + c[..., 2] * r ** 2 + c[..., 3] * r ** 3
        pb = d[..., 0] * r + d[..., 1] * r ** 2 + d[..., 2] * r ** 3
        return kd / pa, kd * pb

    def to_lut(self, template):
        """An ``LS2_LUT``-shaped dict on ``template``'s nodes.

        ``template`` is the published table (``eta``, ``muw``, ``kappa`` are
        copied); ``a`` and ``bb`` are replaced by the model at the nodes.
        """
        eta = np.asarray(template['eta'], float).ravel()
        muw = np.asarray(template['muw'], float).ravel()
        E, M = np.meshgrid(eta, muw, indexing='ij')
        c, d = self.coefficients(E, M)
        return {'eta': np.asarray(template['eta']).copy(),
                'muw': np.asarray(template['muw']).copy(),
                'a': c, 'bb': d,
                'kappa': np.asarray(template['kappa']).copy()}


def fit(rrs, kd, a, bb, eta, muw, *, eta_degree=2, muw_basis=('inv', 'lin'),
        weights=None):
    """Fit both coefficient models by weighted relative least squares.

    Parameters
    ----------
    rrs, kd, a, bb, eta, muw : array_like
        One value per training cell (flattened over scenario, zenith and
        wavelength).  Non-finite cells are dropped.
    eta_degree : int
        Degree of the polynomial in ``sqrt(eta / 0.2)``.
    muw_basis : str or (str, str)
        Key(s) of :data:`MUW_BASES`: one for both tables, or ``(a, bb)``.
    weights : array_like, optional
        Per-cell weights (e.g. to rebalance the ``eta`` axis); default 1.

    Returns
    -------
    SmoothCoefficients
    """
    r, k, a_, b_, e, m = (np.asarray(x, float).ravel()
                          for x in (rrs, kd, a, bb, eta, muw))
    w = np.ones_like(r) if weights is None else np.asarray(weights, float).ravel()
    ok = np.isfinite(r) & np.isfinite(k) & np.isfinite(a_) & np.isfinite(b_) \
        & np.isfinite(e) & np.isfinite(m) & (k > 0) & (a_ > 0) & (b_ > 0) & (w > 0)
    r, k, a_, b_, e, m, w = (x[ok] for x in (r, k, a_, b_, e, m, w))
    bases = (muw_basis, muw_basis) if isinstance(muw_basis, str) else tuple(muw_basis)

    betas = []
    for y, powers, basis in ((k / a_, (0, 1, 2, 3), bases[0]),
                             (b_ / k, (1, 2, 3), bases[1])):
        nm = MUW_BASES[basis](np.ones(1)).shape[-1]
        X = _design(r, e, m, powers, eta_degree, basis)
        sw = np.sqrt(w) / y                  # relative residual, weighted
        beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
        betas.append(beta.reshape(len(powers), eta_degree + 1, nm))

    domain = {'eta': (float(e.min()), float(e.max())),
              'muw': (float(m.min()), float(m.max())),
              'rrs': (float(r.min()), float(r.max())),
              'n_cells': int(r.size)}
    domain['b_over_a'] = None            # filled by the caller if known
    return SmoothCoefficients(betas[0], betas[1], eta_degree, bases,
                              domain=domain)


def save(model: SmoothCoefficients, lut: dict, path):
    """Write the derived table (LS2_LUT keys) plus the model, to ``.npz``."""
    import json
    np.savez(path, eta=lut['eta'], muw=lut['muw'], a=lut['a'], bb=lut['bb'],
             kappa=lut['kappa'], beta_a=model.beta_a, beta_b=model.beta_b,
             eta_degree=np.asarray(model.eta_degree),
             muw_basis=np.asarray(list(model.muw_basis)),
             domain=np.asarray(json.dumps(model.domain)),
             meta=np.asarray(json.dumps(model.meta)))


def load(path):
    """``(LS2_LUT-shaped dict, SmoothCoefficients)`` from :func:`save`'s file."""
    import json
    with np.load(path, allow_pickle=False) as d:
        lut = {k: np.asarray(d[k]) for k in ('eta', 'muw', 'a', 'bb', 'kappa')}
        model = SmoothCoefficients(np.asarray(d['beta_a']), np.asarray(d['beta_b']),
                                   int(d['eta_degree']),
                                   tuple(str(x) for x in d['muw_basis']),
                                   domain=json.loads(str(d['domain'])),
                                   meta=json.loads(str(d['meta'])))
    return lut, model


# --------------------------------------------------------------------------- #
# kappa: the Raman correction (IOPtics ls2 task 13)
# --------------------------------------------------------------------------- #

#: Widening of each row's admissible ``bb/a`` range beyond the training
#: min/max, as a fraction of that range (as ``kd_l23.DOMAIN_TOL``): held-out
#: data legitimately graze the edge, which should not cost a NaN.
KAPPA_RANGE_TOL = 0.01


def fit_kappa(wave, ratio, kappa, *, range_tol=KAPPA_RANGE_TOL):
    """Refit the ``kappa`` table: one cubic in ``bb/a`` per wavelength.

    ``kappa`` is the factor that turns the observed (Raman-inclusive) ``Rrs``
    into the elastic one, ``Rrs_elastic = kappa * Rrs``, so a matched pair of
    elastic and Raman-on simulations of the same water gives it directly as
    their ratio.  The published form (Loisel et al. 2018) is kept: a cubic in
    ``bb/a`` per wavelength, with an admissible ``bb/a`` range per row,
    outside which ``ls2_invert`` reports kappa unavailable.

    Parameters
    ----------
    wave : array_like
        ``(L,)`` wavelengths [nm]; one table row each.
    ratio, kappa : array_like
        ``(N, L)`` true ``bb/a`` and ``kappa`` on ``wave``.  Non-finite cells
        are dropped.
    range_tol : float
        Each row's range is the training min/max widened by this fraction
        of its span.

    Returns
    -------
    numpy.ndarray
        ``(L, 7)`` in the published column order: wavelength, the ``r^3``,
        ``r^2`` and ``r`` coefficients, the constant, then the minimum and
        maximum admissible ``bb/a``.
    """
    wave = np.asarray(wave, float).ravel()
    ratio = np.asarray(ratio, float)
    kappa = np.asarray(kappa, float)
    rows = []
    for j, lam in enumerate(wave):
        x, y = ratio[:, j], kappa[:, j]
        ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
        x, y = x[ok], y[ok]
        p = np.polyfit(x, y, 3)                  # highest power first
        lo, hi = x.min(), x.max()
        pad = range_tol * (hi - lo)
        rows.append([lam, *p, max(lo - pad, 0.0), hi + pad])
    return np.asarray(rows)


def with_kappa(lut, kappa_table):
    """A copy of ``lut`` whose ``kappa`` table is replaced."""
    out = {k: np.asarray(v).copy() for k, v in lut.items()}
    out['kappa'] = np.asarray(kappa_table, float)
    return out


def kappa_eval(kappa_table, wave, ratio):
    """``(kappa, usable)`` from a table, exactly as ``ls2_invert`` evaluates it."""
    from ocpy.ls2.ls2_main import _kappa_terms
    wave = np.broadcast_to(np.asarray(wave, float), np.shape(ratio))
    coef, mins, maxs, in_table = _kappa_terms(wave, np.asarray(kappa_table, float))
    r = np.asarray(ratio, float)
    k = coef[0] * r ** 3 + coef[1] * r ** 2 + coef[2] * r + coef[3]
    usable = in_table & (r >= mins) & (r <= maxs) & np.isfinite(k)
    return k, usable
