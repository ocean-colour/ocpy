=======================
PANGAEA in situ dataset
=======================

The :mod:`ocpy.insitu.pangaea` module provides a small, friendly API for the
**PANGAEA** version-3 compilation of global bio-optical in situ data
(Valente et al. 2022).

* PANGAEA dataset: https://doi.org/10.1594/PANGAEA.941318
* ESSD paper: https://doi.org/10.5194/essd-14-5737-2022

The compilation merges many archives (MOBY, BOUSSOLE, AERONET-OC, SeaBASS,
NOMAD, MERMAID, AMT, ICES, HOT, GeP&CO, ...), spans 1997--2021, and is global
in coverage. It is distributed under CC-BY-4.0.

Overview
========

PANGAEA ships as **seven** tab-separated ``.tab`` files. Each file has a
PANGAEA header block delimited by ``/* ... */``, then a single column-name
line, then the data. Every file shares a global integer observation key
``ID``, so observations can be matched across files.

The seven datasets and their short keys:

============  ================================================  ===================================
Key           Content                                           Filename
============  ================================================  ===================================
``chla``      Chlorophyll-a (HPLC + fluorometric)               ``insitudb_chla_V3.tab``
``rrs``       Remote-sensing reflectance, native wavelengths    ``insitudb_rrs_V3.tab``
``rrs_sat2``  Rrs within +-2 nm of satellite bands              ``insitudb_rrs_satbands2_V3.tab``
``rrs_sat6``  Rrs within +-6 nm of satellite bands              ``insitudb_rrs_satbands6_V3.tab``
``iop``       IOPs (aph, acdom, bbp), Kd, TSM, native           ``insitudb_iopskdtsm_V3.tab``
``iop_sat2``  IOPs/Kd within +-2 nm of satellite bands          ``insitudb_iopskdtsm_satbands2_V3.tab``
``iop_sat6``  IOPs/Kd within +-6 nm of satellite bands          ``insitudb_iopskdtsm_satbands6_V3.tab``
============  ================================================  ===================================

Data location
=============

The data files are large and are **not** packaged with ``ocpy``. The loader
resolves the ``V3`` directory in this order:

#. an explicit ``path=`` argument, then
#. ``$OS_COLOR/PANGAEA/V3``.

The seven files are expected under ``<V3>/datasets/``. If neither candidate
exists, :func:`~ocpy.insitu.pangaea.pangaea_path` raises ``FileNotFoundError``
with the paths it tried.

Loading a dataset
==================

.. code-block:: python

   from ocpy.insitu import pangaea

   # List the available datasets.
   pangaea.file_catalog()

   # Load remote-sensing reflectance (native wavelengths).
   df = pangaea.load('rrs')          # indexed by observation ID

Each :func:`~ocpy.insitu.pangaea.load` call returns a
:class:`pandas.DataFrame`:

* **indexed by** the global observation ``ID``;
* with **friendly column names** -- e.g. ``lat``, ``lon``, ``date_time``,
  ``depth_m``, provenance columns (``dataset`` / ``subdataset`` /
  ``contributor``), quality flags (``qf_time`` / ``qf_chl``), and spectral
  columns such as ``rrs_443`` (native) or ``rrs_OLCI-S3A_band11`` (sat-band);
* with **chlorophyll methods kept separate** (``chla_hplc``, ``chla_fluor``);
* carrying the full per-column metadata in ``df.attrs['columns']`` (original
  PANGAEA name, unit, parsed wavelength, sensor and band), accessible via
  :func:`~ocpy.insitu.pangaea.column_metadata`.

The PANGAEA header length is detected automatically
(:func:`~ocpy.insitu.pangaea.find_header_end`), so no per-file ``skiprows``
is hard-coded.

Extracting a spectrum
=====================

The primary use case is generating an individual spectrum for one
observation. :func:`~ocpy.insitu.pangaea.spectrum` returns a
wavelength-indexed :class:`pandas.Series` with missing wavelengths dropped:

.. code-block:: python

   obs_id = df.index[0]
   spec = pangaea.spectrum(df, obs_id, kind='rrs')

   import matplotlib.pyplot as plt
   plt.plot(spec.index, spec.values)
   plt.xlabel('Wavelength [nm]')
   plt.ylabel('Rrs [1/sr]')

For native-wavelength files the wavelength comes from the column metadata.
For **sat-band** files each value column is paired with a ``Lambda`` column,
and ``spectrum`` uses that **per-row** wavelength automatically.

The ``kind`` argument selects the variable family:

* ``'rrs'`` -- remote-sensing reflectance (``rrs`` files)
* ``'aph'`` -- algal pigment absorption (``iop`` files)
* ``'acdom'`` -- CDOM + detrital absorption
* ``'bbp'`` -- particulate backscatter
* ``'kd'`` -- diffuse attenuation

Long-format reshape
====================

For cross-observation analysis, :func:`~ocpy.insitu.pangaea.to_long`
reshapes one variable family into a tidy frame with columns ``ID``,
``wavelength``, ``value`` and (for sat-band data) ``sensor`` and ``band``:

.. code-block:: python

   long = pangaea.to_long(df, kind='rrs')

API summary
===========

============================================  ===============================================
Function                                      Purpose
============================================  ===============================================
``file_catalog()``                            Map short keys to filenames
``pangaea_path(path=None)``                   Resolve the ``V3`` data directory
``dataset_file(key, path=None)``              Full path to one ``.tab`` file
``find_header_end(filename)``                 Detect the ``*/`` header terminator
``load(key, path=None)``                      Load a dataset into a tidy DataFrame
``spectrum(df, obs_id, kind='rrs')``          One observation's spectrum (Series)
``to_long(df, kind='rrs')``                   Long/tidy reshape of a variable family
``column_metadata(df)``                       Per-column metadata mapping
============================================  ===============================================

See also the demonstration notebook ``nb/PANAGEA/PANAGEA_demo.ipynb`` (kept under
its original directory name; its code uses the former ``panagea`` import).
