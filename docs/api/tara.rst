===========
Tara Oceans
===========

.. module:: ocpy.tara
   :synopsis: Tara Oceans expedition data

The ``tara`` module provides tools for loading and analyzing data from the Tara Oceans
expedition, a global survey of ocean plankton communities.

Overview
--------

The Tara Oceans expedition (2009-2013) collected:

* Bio-optical measurements (absorption, scattering)
* Biogeochemical data (chlorophyll, nutrients)
* Genomic data (metagenomics, metatranscriptomics)
* Physical oceanography (CTD profiles)

Data I/O
--------

.. module:: ocpy.tara.io
   :synopsis: Tara data loading functions

.. autofunction:: ocpy.tara.io.load_db

.. autofunction:: ocpy.tara.io.load_ac_db

.. autofunction:: ocpy.tara.io.load_pg_db

.. autofunction:: ocpy.tara.io.load_tara_umap

.. autofunction:: ocpy.tara.io.load_tara_sequencer

Available Datasets
^^^^^^^^^^^^^^^^^^

**Patrick Gray Database (pg)**

Comprehensive bio-optical dataset with:

* Particulate absorption (ap)
* Phytoplankton absorption (aph)
* CDOM absorption (ag)
* Backscattering coefficients
* Station metadata

**Alison Chase Database (ac)**

Extended analysis including:

* Pigment concentrations
* Particle size distributions
* Community composition estimates

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

   from ocpy.tara import io as tara_io

   # Load Patrick Gray database
   tara = tara_io.load_db(dataset='pg')
   print(f"Number of stations: {len(tara)}")
   print(f"Columns: {tara.columns.tolist()}")

   # Load as GeoDataFrame for spatial analysis
   tara_geo = tara_io.load_pg_db(expedition='all', as_geo=True)

   # Load specific expedition
   tara_polar = tara_io.load_pg_db(expedition='polar')

Data Ingestion
--------------

.. module:: ocpy.tara.ingest
   :synopsis: Tara data ingestion

Functions for reading raw Tara data files.

.. autofunction:: ocpy.tara.ingest.read_one_file

.. autofunction:: ocpy.tara.ingest.load_cruise

.. autofunction:: ocpy.tara.ingest.load_all

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

   from ocpy.tara import ingest

   # Read a single file
   data = ingest.read_one_file('path/to/tara_file.txt')

   # Load entire cruise
   cruise_data = ingest.load_cruise('TARA_001')

   # Load all Tara data
   all_data = ingest.load_all()

Spectra Processing
------------------

.. module:: ocpy.tara.spectra
   :synopsis: Tara spectral data processing

Functions for extracting and processing spectral data.

.. autofunction:: ocpy.tara.spectra.parse_wavelengths

.. autofunction:: ocpy.tara.spectra.spectbl_from_keys

.. autofunction:: ocpy.tara.spectra.spectra_from_table

.. autofunction:: ocpy.tara.spectra.average_spectrum

.. autofunction:: ocpy.tara.spectra.spectrum_from_row

.. autofunction:: ocpy.tara.spectra.single_value

Extracting Spectra
^^^^^^^^^^^^^^^^^^

The Tara database stores spectra as columns with wavelength-based names.
These functions extract spectra into arrays:

.. code-block:: python

   from ocpy.tara import io as tara_io
   from ocpy.tara import spectra

   # Load database
   tara = tara_io.load_db()

   # Extract particulate absorption spectra
   wavelengths, ap_spectra = spectra.spectra_from_table(tara, flavor='ap')
   print(f"Wavelengths: {wavelengths}")
   print(f"Spectra shape: {ap_spectra.shape}")

   # Get spectrum for a single station
   row = tara.iloc[0]
   wave, ap = spectra.spectrum_from_row(row, flavor='ap')

   # Calculate average spectrum
   wave, ap_mean = spectra.average_spectrum(tara, flavor='ap')

Available Spectral Types
^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Tara Spectral Products
   :header-rows: 1
   :widths: 20 30 50

   * - Flavor
     - Variable
     - Description
   * - ap
     - Particulate absorption
     - a_p(λ) in m⁻¹
   * - aph
     - Phytoplankton absorption
     - a_ph(λ) in m⁻¹
   * - ad
     - Detrital absorption
     - a_d(λ) in m⁻¹
   * - ag
     - CDOM absorption
     - a_g(λ) in m⁻¹
   * - bp
     - Particulate scattering
     - b_p(λ) in m⁻¹
   * - bbp
     - Particulate backscattering
     - b_bp(λ) in m⁻¹

Analysis Functions
------------------

.. module:: ocpy.tara.analysis
   :synopsis: Tara data analysis

.. autofunction:: ocpy.tara.analysis.dist_coast

Derived Quantities
------------------

.. module:: ocpy.tara.measures
   :synopsis: Tara derived measurements

Functions for calculating derived biogeochemical quantities.

.. autofunction:: ocpy.tara.measures.chla_boss13

.. autofunction:: ocpy.tara.measures.poc

.. autofunction:: ocpy.tara.measures.add_derived

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

   from ocpy.tara import io as tara_io
   from ocpy.tara import measures

   # Load database
   tara = tara_io.load_db()

   # Add derived quantities
   tara = measures.add_derived(tara, quantities=['chl', 'poc'])
   print(f"New columns: {[c for c in tara.columns if 'derived' in c.lower()]}")

   # Calculate chlorophyll using BOSS method
   chl = measures.chla_boss13(tara)

   # Calculate POC
   poc = measures.poc(tara)

Exploration Tools
-----------------

.. module:: ocpy.tara.explore
   :synopsis: Tara data exploration

Functions for exploring and clustering Tara data.

.. autofunction:: ocpy.tara.explore.prep_spectra

.. autofunction:: ocpy.tara.explore.run_sequencer

Clustering and Dimensionality Reduction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from ocpy.tara import io as tara_io
   from ocpy.tara import explore

   # Load database
   tara = tara_io.load_db()

   # Prepare spectra for analysis
   wavelengths, spectra_clean = explore.prep_spectra(
       wv_grid=None,
       min_sn=1.0
   )

   # Run clustering
   clusters = explore.run_sequencer(wavelengths, spectra_clean)

   # Load pre-computed UMAP projection
   umap_coords = tara_io.load_tara_umap('aph')

Station Metadata
----------------

The Tara database includes rich metadata for each station:

.. list-table:: Tara Metadata Fields
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Description
   * - station
     - Station identifier
   * - lat, lon
     - Geographic coordinates
   * - depth
     - Sampling depth (m)
   * - date
     - Sampling date
   * - cruise
     - Cruise/expedition name
   * - ocean
     - Ocean basin
   * - biome
     - Longhurst biogeochemical province
   * - sst
     - Sea surface temperature
   * - sss
     - Sea surface salinity
   * - chl_sat
     - Satellite-derived chlorophyll

Visualization
-------------

Example visualization of Tara data:

.. code-block:: python

   import matplotlib.pyplot as plt
   import cartopy.crs as ccrs
   from ocpy.tara import io as tara_io

   # Load as GeoDataFrame
   tara = tara_io.load_pg_db(as_geo=True)

   # Create map
   fig, ax = plt.subplots(
       figsize=(12, 6),
       subplot_kw={'projection': ccrs.Robinson()}
   )

   ax.coastlines()
   ax.gridlines(draw_labels=True)

   # Plot stations colored by chlorophyll
   scatter = ax.scatter(
       tara.geometry.x,
       tara.geometry.y,
       c=tara['chl'],
       cmap='viridis',
       transform=ccrs.PlateCarree(),
       norm=plt.matplotlib.colors.LogNorm()
   )
   plt.colorbar(scatter, label='Chl-a (mg/m³)')
   plt.title('Tara Oceans Stations')

Data Access
-----------

The Tara Oceans data requires downloading parquet files:

1. See ``ocpy/data/Tara/README.md`` for download instructions
2. Place files in the ``ocpy/data/Tara/`` directory
3. Files include:

   * ``tara_pg.parquet`` - Patrick Gray database
   * ``tara_ac.feather`` - Alison Chase database

References
----------

* Bricaud, A., et al. (2012). Optical classification of Tara Oceans waters.
  Remote Sensing of Environment, 123, 509-523.

* Boss, E., et al. (2013). The characteristics of particulate absorption,
  scattering and attenuation coefficients in the surface ocean; Contribution
  of the Tara Oceans expedition. Methods in Oceanography, 7, 52-62.

* Tara Oceans Foundation: https://fondationtaraocean.org/en/home/
