=========
Changelog
=========

All notable changes to ocpy will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
------------

Added
^^^^^

* Initial documentation with Sphinx and ReadTheDocs support
* Comprehensive API reference documentation
* User guides for all major modules
* Installation and quickstart guides

[0.1.dev0] - Development
------------------------

This is the initial development release of ocpy.

Features
^^^^^^^^

**Water Optical Properties** (``water/``)

* Pure water absorption coefficients (GSFC and IOCCG 2018 datasets)
* Seawater scattering using Zhang et al. (2009) model
* Refractive index calculations

**LS2 Inversion Model** (``ls2/``)

* LS2 algorithm for IOP retrieval (Loisel et al. 2018)
* Look-up table loading and interpolation
* Kd neural network estimation

**Chlorophyll Algorithms** (``chl/``)

* OC2 band-ratio algorithm
* OC4 band-ratio algorithm

**Inherent Optical Properties** (``iop/``)

* CDOM absorption models (exponential, power-law)
* CDOM fitting functions
* ZLee IOP methods
* Particle cross-sections (Stramski et al. 2001)

**Phytoplankton** (``ph/``)

* Bricaud 1998 absorption model
* Multiple phytoplankton absorption datasets
* Gaussian pigment decomposition
* Chlorophyll-specific absorption fitting

**Satellite Utilities** (``satellites/``)

* PACE wavelength and noise model
* MODIS matchup data and error statistics
* SeaWiFS matchup data and error statistics
* SBG noise model

**PACE Data I/O** (``pace/``)

* Level 2 OC data loading
* Level 2 IOP data loading

**Tara Oceans** (``tara/``)

* Database loading (Patrick Gray and Alison Chase)
* Spectral data extraction
* Derived quantity calculations
* Exploratory analysis tools

**Hydrolight** (``hydrolight/``)

* Loisel+2023 dataset interface
* Chlorophyll calculation from simulations

**Hyper-a Processing** (``hyper_a/``)

* Binary file reading
* Calibration loading
* Full processing workflow
* Temperature and salinity corrections

**Utilities** (``utils/``)

* Coordinate conversions
* JSON I/O
* PCA analysis
* Plotting utilities
* Spectral rebinning

Dependencies
^^^^^^^^^^^^

Core:

* numpy
* scipy
* pandas
* xarray
* matplotlib
* seaborn
* scikit-learn

Optional:

* cartopy (mapping)
* geopandas (geospatial)
* healpy (HEALPix)
* bokeh (interactive plots)
* netCDF4, h5netcdf (file I/O)
* pyarrow (Parquet files)

Known Issues
^^^^^^^^^^^^

* Hydrolight module requires external data download
* Tara module requires external data download
* Some optional dependencies can be difficult to install

Future Plans
^^^^^^^^^^^^

* Expanded algorithm suite
* Additional satellite sensor support
* Machine learning-based algorithms
* Improved validation datasets
* Interactive visualization tools
