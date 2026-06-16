=======
Spectra
=======

.. module:: ocpy.spectra
   :synopsis: Lightweight spectrum classes and array-level helpers

The ``spectra`` package provides two lightweight, numpy-backed containers --
:class:`~ocpy.spectra.core.Spectrum` and
:class:`~ocpy.spectra.core.SpectrumStack` -- together with array-level helper
functions and adapters that build them from the PANAGEA, Loisel+2023, and
Tara Oceans datasets.

Core classes
------------

.. module:: ocpy.spectra.core
   :synopsis: Spectrum and SpectrumStack

.. autoclass:: ocpy.spectra.core.Spectrum
   :members:
   :undoc-members:

.. autoclass:: ocpy.spectra.core.SpectrumStack
   :members:
   :undoc-members:

Array-level helpers
-------------------

.. module:: ocpy.spectra.utils
   :synopsis: Free functions on numpy arrays

These operate on plain numpy arrays (no class dependency) so they can be
reused directly.

.. autofunction:: ocpy.spectra.utils.rebin

.. autofunction:: ocpy.spectra.utils.rebin_to_grid

.. autofunction:: ocpy.spectra.utils.value_at

.. autofunction:: ocpy.spectra.utils.common_grid

.. autofunction:: ocpy.spectra.utils.align_to_grid

Source adapters
---------------

.. module:: ocpy.spectra.io
   :synopsis: Build Spectrum / SpectrumStack from ocpy datasets

.. autofunction:: ocpy.spectra.io.from_panagea

.. autofunction:: ocpy.spectra.io.stack_from_panagea

.. autofunction:: ocpy.spectra.io.from_loisel23

.. autofunction:: ocpy.spectra.io.stack_from_loisel23

.. autofunction:: ocpy.spectra.io.from_tara

.. autofunction:: ocpy.spectra.io.stack_from_tara
