=======================
Spectra (Spectrum API)
=======================

The :mod:`ocpy.spectra` package provides a small, general way to carry an
ocean-colour spectrum (Rrs, *a*, *b*\ :sub:`b`, ...) together with its
provenance, and to operate on it.

Design philosophy
=================

Per the package requirements, a spectrum is a **lightweight container of
basic Python objects** (numpy arrays and scalars), and the real work is done
by **free functions on numpy arrays** rather than by heavy class machinery.
There are two objects:

* :class:`~ocpy.spectra.core.Spectrum` -- a single spectrum.
* :class:`~ocpy.spectra.core.SpectrumStack` -- a collection of spectra,
  backed by a plain list so that *ragged* members (different, or
  different-length, wavelength grids) are always representable.

A :class:`~ocpy.spectra.core.Spectrum` carries:

============  ====================================================
Attribute     Meaning
============  ====================================================
``wavelength``  Wavelengths (nm), 1D numpy array, sorted ascending
``values``      Values, same length as ``wavelength``
``errors``      1-sigma errors (or ``None``)
``units``       Opaque units string, e.g. ``'1/sr'``
``source``      Where the data came from
``date``        Observation date/time (``np.datetime64``)
``lat`` / ``lon``  Latitude / longitude (deg)
``depth``       Depth (m)
``quality``     Named QF flags kept separate, e.g. ``{'qf_time': 0}``
``metadata``    Free-form dict (richer location, sensor/band, ...)
============  ====================================================

Quick start
===========

.. code-block:: python

   import numpy as np
   from ocpy.spectra import Spectrum, SpectrumStack

   wv = np.array([443., 490., 555., 670.])
   rrs = Spectrum(wavelength=wv, values=np.array([5e-3, 4e-3, 2e-3, 7e-4]),
                  units='1/sr', source='example')

   # Interpolate onto a new grid (provenance carried over).
   rrs5 = rrs.rebin(np.arange(440., 680., 5.))

   # Value at a wavelength.
   print(rrs.value_at(500.))

   # Quick plot (matplotlib).
   ax = rrs.plot(marker='o')

Operating on basic arrays
=========================

The array-level helpers in :mod:`ocpy.spectra.utils` take and return plain
numpy arrays:

.. code-block:: python

   from ocpy.spectra import utils

   # Interpolate a single spectrum onto a grid.
   new_vals, new_err = utils.rebin(wv, values, wv_grid, err_vals=err)

   # Bin-average a stack, shape (nspec, nwave) -> (nspec, nbin).
   rwave, rvals, rerr = utils.rebin_to_grid(wv, values_2d, errs_2d, edges)

   # Rebin a ragged collection onto one common grid.
   vals_2d, err_2d = utils.align_to_grid(list_of_wv, list_of_values, wv_grid)

Stacks: gridded vs. ragged
==========================

A :class:`~ocpy.spectra.core.SpectrumStack` is *gridded* when all members
share one wavelength grid. Only then can it expose a 2D array view:

.. code-block:: python

   stack = SpectrumStack([rrs_a, rrs_b])
   if stack.is_gridded:
       wv, values, errors = stack.as_array()   # (nwave,), (nspec, nwave)

   # Ragged stacks must be rebinned onto a common grid first.
   gridded = stack.rebin(np.arange(440., 680., 5.))

Per-spectrum provenance is available as arrays: ``stack.dates``,
``stack.lats``, ``stack.lons``, ``stack.depths``, ``stack.sources``.

Building spectra from datasets
==============================

The adapters in :mod:`ocpy.spectra.io` map the existing dataset loaders into
the classes:

.. code-block:: python

   from ocpy.spectra import io as spectra_io
   from ocpy.insitu import panagea
   from ocpy.hydrolight import loisel23

   # PANAGEA
   df = panagea.load('rrs')
   rrs = spectra_io.from_panagea(df, df.index[0], kind='rrs')

   # Loisel+2023 (one Lambda grid; units passed explicitly)
   ds = loisel23.load_ds(1, 0)
   rrs_l23 = spectra_io.from_loisel23(ds, 0, var='Rrs', units='1/sr')
   stack = spectra_io.stack_from_loisel23(ds, var='Rrs', indices=range(50))

   # Tara Oceans (ap / cp)
   # spec = spectra_io.from_tara(row, flavor='ap')

xarray and netCDF interop
=========================

A spectrum converts to an :class:`xarray.Dataset` in which **wavelength is a
data variable, not a coordinate**. The dataset uses a bare index dimension
``channel`` carrying three data variables -- ``wavelength``, ``values`` and
``errors`` -- with provenance stored in ``.attrs`` (the ``date`` as an
ISO-8601 string, ``quality`` / ``metadata`` as JSON).

.. code-block:: python

   ds = rrs.to_xarray()
   assert 'wavelength' in ds.data_vars     # not a coordinate
   rrs2 = Spectrum.from_xarray(ds)

   # netCDF round trip
   rrs.to_netcdf('rrs.nc')
   rrs3 = Spectrum.read_netcdf('rrs.nc')

For a :class:`~ocpy.spectra.core.SpectrumStack`, ``to_xarray`` produces
``(spectrum, channel)`` data variables; ragged members are NaN-padded to the
longest row, and the per-row ``wavelength`` variable records the true grid so
the pad is unambiguous on read. Per-spectrum provenance is stored as a JSON
list in ``ds.attrs['provenance_json']`` (use the array accessors above to
slice it).

See also
========

* :doc:`api/spectra` -- full API reference.
* The demo notebook ``nb/Spectra/Spectra_demo.ipynb``.
