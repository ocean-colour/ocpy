"""``<Kd>_1`` networks trained on the L23 Hydrolight corpus (IOPtics ls2 task 11).

Two networks, trained in IOPtics (``ioptics.kd_net``, Flax/Optax) and shipped
here as plain weights so that evaluating them needs NumPy only:

``'L23_hyper_v1'``
    ``Rrs`` at every L23 band from 400 to 750 nm (71 inputs, 5 nm), the
    LS2-PACE deliverable.  Feed it OCI ``Rrs`` interpolated onto that grid.
    **It has no in-situ validation**: no dataset on hand pairs hyperspectral
    ``Rrs`` with measured Kd, so it is validated against held-out L23 only.
``'L23_seawifs_v1'``
    ``Rrs`` at 443, 490, 510, 555 and 670 nm, the SeaWiFS-like sibling that
    exists so there is a real-data check (PANGAEA).

Both map ``Rrs`` to ``ln(mu_w * <Kd>_1)`` at the 71 L23 wavelengths
400-750 nm, where ``mu_w`` is the cosine of the refracted solar beam
(Snell, ``n_w = 1.34``, as the authors' networks use).  L23's ``<Kd>_1``
scales as ``1/mu_w`` to within about 1% across scenarios, so the network
learns a nearly geometry-free quantity and :func:`kd_l23` multiplies the
analytic ``1/mu_w`` back.  Whether ``mu_w`` is *also* an input is recorded in
the weights file (``geometry_input``).

Scope, stated in the weights and enforced here:

- **Clear water only.**  L23 has Kd(490) <= 0.65 m^-1 and 95% below 0.10, so
  the authors' turbid branch is not attempted.  Inputs outside the trained
  range of any feature (by more than 1% of its span) are flagged
  ``out_of_domain``; the value is still returned.
- **Solar zenith.**  L23 has 0, 30 and 60 degrees.  Up to 60 degrees is
  interpolation.  60-70 degrees is accepted but flagged ``extrapolated_sza``:
  with the ``1/mu_w`` factor analytic, that extrapolation is arithmetic, not
  learned (and with ``geometry_input`` False the network never sees the angle
  at all).  Beyond 70 degrees the result is NaN, flagged ``unsupported_sza``.

Weights live in ``ocpy/data/LS2/Kd_L23_<name>.npz`` with their standardisation,
trained domain, feature names, envelope and provenance; :func:`load_network`
refuses a file whose feature list does not match its own band list.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources

import numpy as np

#: Refractive index of seawater for the refracted solar beam (as ``kd_nn``).
N_WATER = 1.34

#: Networks shipped, by name -> weights file under ``ocpy/data/LS2``.
NETWORKS = {'L23_hyper_v1': 'Kd_L23_hyper_v1.npz',
            'L23_seawifs_v1': 'Kd_L23_seawifs_v1.npz'}

#: How far beyond its trained range a feature must go before it is flagged,
#: as a fraction of that range (the convention of ``robust.rt.emulator``).
DOMAIN_TOL = 0.01


def muw(sza):
    """Cosine of the refracted solar beam just beneath the surface."""
    s = np.deg2rad(np.asarray(sza, dtype=float))
    return np.cos(np.arcsin(np.sin(s) / N_WATER))


@dataclass(frozen=True)
class KdL23Network:
    """A loaded network: weights, standardisation, domain and envelope.

    Attributes
    ----------
    name : str
    layers : tuple of (W, b)
        ``W`` is ``(n_in, n_out)``; hidden layers use ``tanh``, the last is
        linear.
    x_mean, x_std : numpy.ndarray
        Input standardisation, from the training split only.
    y_mean, y_std : numpy.ndarray
        Output standardisation of ``ln(mu_w <Kd>_1)``, per output wavelength.
    bands : numpy.ndarray
        Input ``Rrs`` wavelengths [nm], in order.
    out_wave : numpy.ndarray
        Output wavelengths [nm].
    geometry_input : bool
        Whether ``mu_w`` is the last input feature.
    domain : numpy.ndarray
        ``(2, n_features)`` trained min/max of the raw features.
    theta_s_trained, theta_s_sanctioned : tuple of float
        Solar-zenith span of the training data, and the span accepted
        (beyond it: NaN).
    meta : dict
        Provenance (training data, split, seed, commit, scores).
    """

    name: str
    layers: tuple
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    bands: np.ndarray
    out_wave: np.ndarray
    geometry_input: bool
    domain: np.ndarray
    theta_s_trained: tuple
    theta_s_sanctioned: tuple
    meta: dict = field(default_factory=dict)

    @property
    def features(self) -> tuple:
        """Feature names, in input order."""
        names = tuple(f'Rrs_{b:g}' for b in self.bands)
        return names + (('muw',) if self.geometry_input else ())

    def forward(self, x_raw):
        """``ln(mu_w <Kd>_1)`` at :attr:`out_wave` from raw features ``(N, F)``."""
        h = (np.asarray(x_raw, dtype=float) - self.x_mean) / self.x_std
        for i, (w, b) in enumerate(self.layers):
            h = h @ w + b
            if i < len(self.layers) - 1:
                h = np.tanh(h)
        return h * self.y_std + self.y_mean


def save_network(net: KdL23Network, path):
    """Write a network to ``.npz`` (the format :func:`from_npz` reads)."""
    arrays = {f'layer{i}_W': np.asarray(w) for i, (w, _) in enumerate(net.layers)}
    arrays.update({f'layer{i}_b': np.asarray(b)
                   for i, (_, b) in enumerate(net.layers)})
    arrays.update(
        n_layers=np.asarray(len(net.layers)), x_mean=net.x_mean,
        x_std=net.x_std, y_mean=net.y_mean, y_std=net.y_std, bands=net.bands,
        out_wave=net.out_wave, geometry_input=np.asarray(net.geometry_input),
        domain=net.domain, features=np.asarray(net.features),
        theta_s_trained=np.asarray(net.theta_s_trained, dtype=float),
        theta_s_sanctioned=np.asarray(net.theta_s_sanctioned, dtype=float),
        name=np.asarray(net.name), meta=np.asarray(json.dumps(net.meta)))
    np.savez(path, **arrays)


def from_npz(path) -> KdL23Network:
    """Read a network written by :func:`save_network`.

    Raises
    ------
    ValueError
        If the stored feature names do not match the stored bands and
        ``geometry_input`` -- the weights would run and return plausible
        nonsense, so this is a refusal.
    """
    with np.load(path, allow_pickle=False) as d:
        n = int(d['n_layers'])
        layers = tuple((np.asarray(d[f'layer{i}_W'], dtype=float),
                        np.asarray(d[f'layer{i}_b'], dtype=float))
                       for i in range(n))
        net = KdL23Network(
            name=str(d['name']), layers=layers,
            x_mean=np.asarray(d['x_mean'], float), x_std=np.asarray(d['x_std'], float),
            y_mean=np.asarray(d['y_mean'], float), y_std=np.asarray(d['y_std'], float),
            bands=np.asarray(d['bands'], float), out_wave=np.asarray(d['out_wave'], float),
            geometry_input=bool(d['geometry_input']),
            domain=np.asarray(d['domain'], float),
            theta_s_trained=tuple(float(v) for v in d['theta_s_trained']),
            theta_s_sanctioned=tuple(float(v) for v in d['theta_s_sanctioned']),
            meta=json.loads(str(d['meta'])))
        stored = tuple(str(f) for f in d['features'])
    if stored != net.features:
        raise ValueError(f'{path}: stored features {stored[:3]}... do not match '
                         f'the network\'s own {net.features[:3]}...; retrain '
                         'rather than reinterpret the weights')
    return net


@lru_cache(maxsize=None)
def load_network(name: str = 'L23_hyper_v1') -> KdL23Network:
    """Load a shipped network by name (cached: the file is read once)."""
    if name not in NETWORKS:
        raise ValueError(f'Unknown L23 Kd network {name!r}; choose from '
                         f'{sorted(NETWORKS)}')
    path = os.path.join(resources.files('ocpy'), 'data', 'LS2', NETWORKS[name])
    return from_npz(path)


def kd_l23(Rrs, sza, wave=None, network: str | KdL23Network = 'L23_hyper_v1', *,
           return_flags: bool = False):
    """``<Kd>_1`` from ``Rrs`` and solar zenith with an L23-trained network.

    Parameters
    ----------
    Rrs : array_like
        Remote-sensing reflectance [sr^-1] at the network's ``bands``,
        ``(N, nb)`` or ``(nb,)``.  Negative values (noise in the red) are
        accepted: the networks were trained on noisy spectra.
    sza : array_like
        Solar zenith [deg], ``(N,)`` or scalar.
    wave : array_like, optional
        Output wavelengths [nm].  Default: the network's own ``out_wave``
        (400-750 nm, 5 nm).  Other values are interpolated in ``ln Kd``;
        outside ``out_wave`` they are NaN.
    network : str or KdL23Network
        A name from :data:`NETWORKS`, or a loaded network.
    return_flags : bool
        Also return per-spectrum flags ``out_of_domain``, ``extrapolated_sza``
        and ``unsupported_sza``.

    Returns
    -------
    Kd : numpy.ndarray
        ``(N, L)`` [m^-1].
    flags : dict, optional
    """
    net = load_network(network) if isinstance(network, str) else network
    rrs = np.atleast_2d(np.asarray(Rrs, dtype=float))
    if rrs.shape[-1] != net.bands.size:
        raise ValueError(f'{net.name} expects Rrs at {net.bands.size} bands, '
                         f'got shape {rrs.shape}')
    n = rrs.shape[0]
    sza = np.broadcast_to(np.asarray(sza, dtype=float).ravel(), (n,)) \
        if np.size(sza) in (1, n) else None
    if sza is None:
        raise ValueError('sza must be a scalar or have one value per spectrum')
    mu = muw(sza)
    x = np.column_stack([rrs, mu]) if net.geometry_input else rrs
    kd = np.exp(net.forward(x)) / mu[:, None]

    lo, hi = net.theta_s_sanctioned
    unsupported = (sza < lo) | (sza > hi) | ~np.isfinite(sza)
    kd[unsupported] = np.nan
    if wave is not None:
        w = np.atleast_1d(np.asarray(wave, dtype=float))
        lk = np.log(kd)
        kd = np.stack([np.exp(np.interp(w, net.out_wave, row, left=np.nan,
                                        right=np.nan)) for row in lk])
    if not return_flags:
        return kd
    span = net.domain[1] - net.domain[0]
    tol = DOMAIN_TOL * np.where(span > 0, span, 1.0)
    ood = np.any((x < net.domain[0] - tol) | (x > net.domain[1] + tol), axis=1)
    tlo, thi = net.theta_s_trained
    extrap = ~unsupported & ((sza < tlo) | (sza > thi))
    return kd, {'out_of_domain': ood, 'extrapolated_sza': extrap,
                'unsupported_sza': unsupported}
