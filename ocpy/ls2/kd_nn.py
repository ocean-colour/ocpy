"""Neural networks for the diffuse attenuation coefficient ``<Kd>_1``.

LS2 needs ``<Kd>_1``, the average attenuation coefficient of downwelling
planar irradiance between the surface and the first attenuation depth.  In
remote-sensing use it comes from a neural network fed by ``Rrs`` and the solar
zenith angle.  This module ports the networks of the SIO Ocean Optics Research
Laboratory's ``Kd_NN_Distribution`` (Kehrli, Taylor, Reynolds, Stramski and
Loisel; MIT licence), which follow Jamet et al. (2012).

Three releases are carried, selected by name:

``'MODIS_v1.1'``
    The authors' MODIS network as of 2023-10-10 (commit ``a5c7ec2``), and the
    one ocpy has always shipped (``weights_1/2.csv``).  Clear-water hidden
    layers 8/6, turbid 9/6; inputs ``[Rrs, lambda, muw]``.  It reproduces the
    authors' v1.1 reference vector to machine precision.  It was the default
    until 2026-10-04, when the LS2 Kd diagnostic (IOPtics ls2 task 10) found
    it 15-19% high against L23's ``<Kd>_1`` at 410-490 nm and swinging from
    -22% to +25% across 400-700 nm.  Pass it by name to reproduce earlier
    results.
``'MODIS_v1.3'``
    The authors' current MODIS network (2025-04-15).  A retrained network,
    not a bug fix: clear 8/8, turbid 4/4, inputs reordered to
    ``[Rrs, muw, lambda]``.  On the 100 reference spectra it differs from
    v1.1 by up to tens of percent.  The default of :func:`kd_nn` and
    :func:`Kd_NN_MODIS` since 2026-10-04: against L23 it reads within 3.2%
    (median over 400-700 nm), but still +15% at 490 nm.
``'PACE_v2.3'``
    The authors' PACE network: ``Rrs`` at 12 PACE wavelengths, ``sza`` (not
    ``muw``) as an input, a plain ``tanh`` activation, clear 19/17 and
    turbid 17/9.

All three switch between a clear- and a turbid-water network on a blue/green
``Rrs`` ratio of 0.85 (488/547 for MODIS, 490/560 for PACE), and return NaN
where any ``Rrs`` input to the selected branch is negative.

:func:`kd_nn` is the vectorized entry point: ``(N, nb)`` spectra by ``(L,)``
output wavelengths in one call, weights loaded once and cached.
:func:`Kd_NN_MODIS` and :func:`Kd_NN_PACE` are scalar wrappers with the
authors' single-spectrum, single-wavelength contract; they always return a
``(1, 1)`` array, NaN included.

References
----------
Jamet, C., H. Loisel and D. Dessailly (2012), Retrieval of the spectral
diffuse attenuation coefficient Kd(lambda) in open and coastal ocean waters
using a neural network inversion, *J. Geophys. Res.*, 117, C10023.

Loisel, H. et al. (2018), An inverse model for estimating the optical
absorption and backscattering coefficients of seawater from remote-sensing
reflectance over a broad range of oceanic and coastal marine environments,
*J. Geophys. Res. Oceans*, 123, 2141-2171.
"""

from __future__ import annotations

import functools
import os
import warnings
from dataclasses import dataclass
from importlib.resources import files

import numpy as np

from ocpy.ls2 import io as ls2_io

#: Refractive index of seawater used to refract the solar beam.
N_WATER = 1.34

#: Blue/green ``Rrs`` ratio at and above which the clear-water network is used.
CLEAR_RATIO = 0.85


@dataclass(frozen=True)
class _Branch:
    """How one branch (clear or turbid) of a network is wired.

    Attributes
    ----------
    rrs_idx : tuple of int
        Columns of the input ``Rrs`` fed to this branch, in order.
    aux : tuple of str
        The non-``Rrs`` inputs that follow, in order, from ``'muw'``,
        ``'sza'`` and ``'lambda'``.
    stat_rows : tuple of int
        Rows of ``train_switch`` holding the mean and standard deviation of
        each input, followed by the row of the output (``log10 Kd``).
    arch : tuple of int
        ``(ne, nc1, nc2)``: input neurons and the two hidden-layer widths.
    """

    rrs_idx: tuple
    aux: tuple
    stat_rows: tuple
    arch: tuple


@dataclass(frozen=True)
class _Spec:
    """Static description of one released network, transcribed from its ``.m``."""

    name: str
    bands: tuple
    switch: tuple       # (numerator, denominator) columns of Rrs
    clear: _Branch
    turbid: _Branch
    activation: str     # 'lecun' (1.715905 tanh(2x/3)) or 'tansig' (tanh)


_SPECS = {
    'MODIS_v1.1': _Spec(
        name='MODIS_v1.1', bands=(443., 488., 531., 547., 667.), switch=(1, 3),
        # Kd_NN_MODIS.m @ a5c7ec2: inputs = [Rrs, lambda, muw]
        clear=_Branch((0, 1, 2, 3), ('lambda', 'muw'),
                      (1, 2, 3, 4, 6, 7, 8), (6, 8, 6)),
        turbid=_Branch((0, 1, 2, 3, 4), ('lambda', 'muw'),
                       (1, 2, 3, 4, 5, 6, 7, 8), (7, 9, 6)),
        activation='lecun'),
    'MODIS_v1.3': _Spec(
        name='MODIS_v1.3', bands=(443., 488., 531., 547., 667.), switch=(1, 3),
        # Kd_NN_MODIS.m @ 19c2501: inputs = [Rrs, muw, lambda]
        clear=_Branch((0, 1, 2, 3), ('muw', 'lambda'),
                      (1, 2, 3, 4, 7, 6, 8), (6, 8, 8)),
        turbid=_Branch((0, 1, 2, 3, 4), ('muw', 'lambda'),
                       (1, 2, 3, 4, 5, 7, 6, 8), (7, 4, 4)),
        activation='lecun'),
    'PACE_v2.3': _Spec(
        name='PACE_v2.3',
        bands=(440., 470., 490., 510., 530., 560., 580., 600., 620., 640.,
               670., 700.),
        switch=(2, 5),
        # Kd_NN_PACE.m @ 982b52c: inputs = [Rrs, sza, lambda]; the clear
        # branch drops Rrs(670) and Rrs(700).
        clear=_Branch(tuple(range(10)), ('sza', 'lambda'),
                      tuple(range(10)) + (12, 13, 14), (12, 19, 17)),
        turbid=_Branch(tuple(range(12)), ('sza', 'lambda'),
                       tuple(range(15)), (14, 17, 9)),
        activation='tansig'),
}

#: Names accepted by :func:`kd_nn` and :func:`load_network`.
NETWORKS = tuple(_SPECS)


@dataclass(frozen=True)
class _Layers:
    """Weights, biases and normalization of one branch, ready to evaluate."""

    w1: np.ndarray      # (nc1, ne)
    b1: np.ndarray      # (nc1,)
    w2: np.ndarray      # (nc2, nc1)
    b2: np.ndarray      # (nc2,)
    wout: np.ndarray    # (nc2,)
    bout: float
    mu: np.ndarray      # (ne,) input means
    std: np.ndarray     # (ne,) input standard deviations
    mu_kd: float        # output mean, log10 Kd
    std_kd: float       # output standard deviation, log10 Kd


@dataclass(frozen=True)
class KdNetwork:
    """A loaded Kd network: its static description plus both branches."""

    spec: _Spec
    clear: _Layers
    turbid: _Layers
    version: str
    source_commit: str

    @property
    def bands(self) -> tuple:
        """Input ``Rrs`` wavelengths [nm], in the order expected."""
        return self.spec.bands


def _layers(npz, prefix, branch, mean, std):
    """Reshape one branch's LUT columns exactly as MATLAB ``reshape`` does."""
    ne, nc1, nc2 = branch.arch
    col = {k: np.asarray(npz[f'{prefix}_{k}'], dtype=float)
           for k in ('b1', 'b2', 'bout', 'w1', 'w2', 'wout')}
    expected = {'b1': nc1, 'b2': nc2, 'bout': 1, 'w1': nc1 * ne,
                'w2': nc2 * nc1, 'wout': nc2}
    for key, size in expected.items():
        if col[key].size != size:
            raise ValueError(f'{prefix}_{key} has {col[key].size} values, '
                             f'expected {size} for architecture {branch.arch}')
    rows = np.asarray(branch.stat_rows)
    return _Layers(
        # MATLAB reshape is column-major.
        w1=col['w1'].reshape((nc1, ne), order='F'),
        b1=col['b1'],
        w2=col['w2'].reshape((nc2, nc1), order='F'),
        b2=col['b2'],
        wout=col['wout'],
        bout=float(col['bout'][0]),
        mu=mean[rows[:-1]], std=std[rows[:-1]],
        mu_kd=float(mean[rows[-1]]), std_kd=float(std[rows[-1]]))


@functools.lru_cache(maxsize=None)
def load_network(name: str = 'MODIS_v1.1') -> KdNetwork:
    """Load (once, then from cache) one of the released Kd networks.

    Parameters
    ----------
    name : str, optional
        One of :data:`NETWORKS`.  Default ``'MODIS_v1.3'`` (was
        ``'MODIS_v1.1'`` before 2026-10-04).

    Returns
    -------
    KdNetwork
    """
    if name not in _SPECS:
        raise ValueError(f'Unknown Kd network {name!r}; choose from {NETWORKS}')
    spec = _SPECS[name]
    path = files('ocpy').joinpath(
        os.path.join('data', 'LS2', f'Kd_NN_LUT_{name}.npz'))
    with np.load(path) as npz:
        mean = np.asarray(npz['train_mean'], dtype=float)
        std = np.asarray(npz['train_std'], dtype=float)
        return KdNetwork(spec=spec,
                         clear=_layers(npz, 'clear', spec.clear, mean, std),
                         turbid=_layers(npz, 'turbid', spec.turbid, mean, std),
                         version=str(npz['version']),
                         source_commit=str(npz['source_commit']))


def _activate(z, kind):
    """The hidden-layer transfer function, exactly as each ``.m`` writes it."""
    if kind == 'tansig':
        # MATLAB's tansig: 2/(1+exp(-2n))-1, which is tanh(n) to rounding.
        with np.errstate(over='ignore'):
            return 2. / (1. + np.exp(-2. * z)) - 1.
    raise AssertionError(kind)


def _forward(x_n, layers, kind):
    """Evaluate one branch on normalized inputs ``x_n`` of shape ``(M, ne)``."""
    if kind == 'lecun':
        # The MODIS .m files use 0.6666667 in the first layer and 2./3 in
        # the second; kept verbatim for bit-level agreement.
        a = 1.715905 * np.tanh(0.6666667 * (x_n @ layers.w1.T + layers.b1))
        b = 1.715905 * np.tanh((2. / 3.) * (a @ layers.w2.T + layers.b2))
    else:
        a = _activate(x_n @ layers.w1.T + layers.b1, kind)
        b = _activate(a @ layers.w2.T + layers.b2, kind)
    y = b @ layers.wout + layers.bout
    return 10.0 ** (1.5 * y * layers.std_kd + layers.mu_kd)


def kd_nn(Rrs, sza, wave, network: str = 'MODIS_v1.3', *,
          return_branch: bool = False):
    """Vectorized ``<Kd>_1`` from ``Rrs`` and solar zenith angle.

    Parameters
    ----------
    Rrs : array_like
        Remote-sensing reflectance [sr^-1] at the network's bands
        (``load_network(network).bands``), shape ``(N, nb)`` or ``(nb,)``.
    sza : array_like
        Solar zenith angle [deg], shape ``(N,)`` or scalar.
    wave : array_like
        Output wavelengths [nm] at which Kd is wanted, shape ``(L,)`` or
        scalar.  Wavelength is an *input* to these networks, so any value
        works numerically; the training data span roughly 350-750 nm.
    network : str, optional
        One of :data:`NETWORKS`.  Default ``'MODIS_v1.3'`` (was
        ``'MODIS_v1.1'`` before 2026-10-04).
    return_branch : bool, optional
        Also return a dict of per-spectrum boolean arrays ``clear``,
        ``turbid`` and ``negative``.  A spectrum whose switch ratio is not
        finite is in neither branch and comes back NaN.

    Returns
    -------
    Kd : numpy.ndarray
        Average attenuation coefficient [m^-1] between the surface and the
        first attenuation depth, shape ``(N, L)``.  NaN where an ``Rrs``
        input to the selected branch is negative; no warning is emitted.
    branch : dict, optional
        Only if ``return_branch``.
    """
    net = load_network(network)
    spec = net.spec

    Rrs = np.atleast_2d(np.asarray(Rrs, dtype=float))
    if Rrs.shape[-1] != len(spec.bands):
        raise ValueError(f'{network} expects Rrs at {len(spec.bands)} bands '
                         f'{spec.bands}, got shape {Rrs.shape}')
    n = Rrs.shape[0]
    sza = np.broadcast_to(np.asarray(sza, dtype=float).ravel(), (n,)) \
        if np.size(sza) in (1, n) else None
    if sza is None:
        raise ValueError('sza must be a scalar or have one value per spectrum')
    wave = np.atleast_1d(np.asarray(wave, dtype=float)).ravel()
    nl = wave.size

    muw = np.cos(np.arcsin(np.sin(np.deg2rad(sza)) / N_WATER))
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = Rrs[:, spec.switch[0]] / Rrs[:, spec.switch[1]]
    clear = ratio >= CLEAR_RATIO
    turbid = ratio < CLEAR_RATIO

    Kd = np.full((n, nl), np.nan)
    negative = np.zeros(n, dtype=bool)
    for mask, branch, layers in ((clear, spec.clear, net.clear),
                                 (turbid, spec.turbid, net.turbid)):
        rrs = Rrs[:, branch.rrs_idx]
        neg = mask & np.any(rrs < 0., axis=1)
        negative |= neg
        use = mask & ~neg
        if not use.any():
            continue
        m = int(use.sum())
        aux = {'muw': np.repeat(muw[use], nl), 'sza': np.repeat(sza[use], nl),
               'lambda': np.tile(wave, m)}
        x = np.column_stack([np.repeat(rrs[use], nl, axis=0)]
                            + [aux[name] for name in branch.aux])
        x_n = (2. / 3.) * (x - layers.mu) / layers.std
        Kd[use] = _forward(x_n, layers, spec.activation).reshape(m, nl)

    if return_branch:
        return Kd, {'clear': clear, 'turbid': turbid, 'negative': negative}
    return Kd


def _scalar(network, Rrs, sza, lambda_):
    """Shared body of the scalar wrappers: one spectrum, one wavelength."""
    Kd, info = kd_nn(np.asarray(Rrs, dtype=float).reshape(1, -1), sza,
                     lambda_, network, return_branch=True)
    if info['negative'][0]:
        warnings.warn('Negative Rrs input detected. Kd set to NaN.')
    return Kd[:, :1]


def Kd_NN_MODIS(Rrs, sza, lambda_, *, version: str = '1.3'):
    """``<Kd>_1`` at one wavelength from MODIS ``Rrs``; scalar wrapper.

    A thin wrapper over :func:`kd_nn` with the authors' ``Kd_NN_MODIS.m``
    contract.  The band list (443, 488, 531, 547, 667 nm) and the "40,000
    inputs and outputs" of the training set are as stated in the authors'
    own docstring; neither is described in Loisel et al. (2018).

    Parameters
    ----------
    Rrs : array_like
        Spectral remote-sensing reflectance [sr^-1] at 443, 488, 531, 547
        and 667 nm, five values.
    sza : float
        Solar zenith angle [deg].
    lambda_ : float
        Output wavelength [nm].
    version : str, optional
        ``'1.3'`` (default since 2026-10-04; the authors' current release)
        or ``'1.1'`` (the network ocpy shipped before then).

    Returns
    -------
    numpy.ndarray
        Kd [m^-1], always shape ``(1, 1)``.  NaN, with a warning, if an
        ``Rrs`` input to the selected branch is negative.
    """
    return _scalar(f'MODIS_v{version}', Rrs, sza, lambda_)


def Kd_NN_PACE(Rrs, sza, lambda_):
    """``<Kd>_1`` at one wavelength from PACE ``Rrs``; scalar wrapper.

    A thin wrapper over :func:`kd_nn` with the authors' ``Kd_NN_PACE.m`` (v2.3)
    contract.

    Parameters
    ----------
    Rrs : array_like
        Spectral remote-sensing reflectance [sr^-1] at 440, 470, 490, 510,
        530, 560, 580, 600, 620, 640, 670 and 700 nm, twelve values.
    sza : float
        Solar zenith angle [deg].
    lambda_ : float
        Output wavelength [nm].

    Returns
    -------
    numpy.ndarray
        Kd [m^-1], always shape ``(1, 1)``.  NaN, with a warning, if an
        ``Rrs`` input to the selected branch is negative.
    """
    return _scalar('PACE_v2.3', Rrs, sza, lambda_)


def load_weights(wtype: str):
    """Load the v1.1 MODIS weights in the historical layout (legacy API).

    Kept for backward compatibility; :func:`kd_nn` does not use it.  The
    matrices are transposed relative to :class:`KdNetwork` -- ``w1`` is
    ``(ne, nc1)`` -- because :func:`MLP_Kd` multiplies on the right.  The
    old "these could be backwards" doubt about this reshape is settled: it
    is the transpose of MATLAB's column-major ``reshape(w1, nc1, ne)``, and
    the v1.1 network reproduces the authors' reference vector.

    Parameters
    ----------
    wtype : str
        ``'turbid'`` for the turbid-water network; anything else gives the
        clear-water network.

    Returns
    -------
    tuple
        ``(w1, b1, w2, b2, wout, bout)``.
    """
    weights_1, weights_2, _ = ls2_io.load_Kd_tables()

    # Input neurons, first and second hidden layers, outputs.
    if wtype == 'turbid':
        ne, nc1, nc2, ns = 7, 9, 6, 1
        weights = weights_2
    else:
        ne, nc1, nc2, ns = 6, 8, 6, 1
        weights = weights_1

    b1, b2, bout = weights["b1"], weights["b2"], weights["bout"]
    w1, w2, wout = weights["w1"], weights["w2"], weights["wout"]

    b1 = b1[np.isfinite(b1)].values
    b2 = b2[np.isfinite(b2)].values
    bout = bout[np.isfinite(bout)].values
    w1 = w1[np.isfinite(w1)].values
    w2 = w2[np.isfinite(w2)].values
    wout = wout[np.isfinite(wout)].values

    w1 = w1.reshape(ne, nc1)
    w2 = w2.reshape(nc1, nc2)
    wout = wout.reshape(ns, nc2)

    return w1, b1, w2, b2, wout, bout


def MLP_Kd(x, w1, b1, w2, b2, wout, bout, muKd, stdKd):
    """Forward pass of the v1.1 MODIS network in the legacy layout.

    Kept for backward compatibility with :func:`load_weights`; :func:`kd_nn`
    does not use it.

    Parameters
    ----------
    x : numpy.ndarray
        Normalized inputs, shape ``(rx, ne)``.
    w1, b1, w2, b2, wout, bout : numpy.ndarray
        As returned by :func:`load_weights`.
    muKd, stdKd : float or numpy.ndarray
        Mean and standard deviation of the training ``log10 Kd``.

    Returns
    -------
    numpy.ndarray
        Kd [m^-1], shape ``(rx, 1)``.
    """
    rx, _ = x.shape
    a = 1.715905 * np.tanh(0.6666667 * (np.dot(x, w1) + np.ones((1, rx)) * b1))
    b = 1.715905 * np.tanh((2.0 / 3.0) * (np.dot(a, w2) + np.ones((1, rx)) * b2))
    y = np.dot(b, wout.T) + bout * np.ones((rx, 1))
    return 10.0 ** (1.5 * y * stdKd + muKd)
