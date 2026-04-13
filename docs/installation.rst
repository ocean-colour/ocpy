============
Installation
============

Requirements
------------

ocpy requires Python 3.8 or later. The package has been tested on Linux, macOS, and Windows.

Core Dependencies
^^^^^^^^^^^^^^^^^

The following packages are required:

* **NumPy** (>=1.20): Array operations and numerical computing
* **SciPy** (>=1.7): Scientific computing and interpolation
* **pandas** (>=1.3): Data manipulation and analysis
* **xarray** (>=0.19): N-dimensional labeled arrays (for netCDF data)
* **matplotlib** (>=3.4): Plotting and visualization
* **seaborn** (>=0.11): Statistical visualization
* **scikit-learn** (>=0.24): Machine learning (for PCA, clustering)

Optional Dependencies
^^^^^^^^^^^^^^^^^^^^^

For full functionality, the following packages are recommended:

* **cartopy** (>=0.20): Geospatial plotting and map projections
* **geopandas** (>=0.10): Geospatial data handling
* **healpy** (>=1.15): HEALPix spherical pixelization
* **bokeh** (>=2.4): Interactive visualization
* **h5netcdf** (>=1.0): HDF5/netCDF file I/O
* **netCDF4** (>=1.5): NetCDF file I/O
* **pyarrow** (>=6.0): Parquet file support (for Tara data)
* **openpyxl** (>=3.0): Excel file support
* **geopy** (>=2.2): Geocoding and distance calculations
* **pyproj** (>=3.2): Cartographic projections

Installation Methods
--------------------

From PyPI (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^

Once published, install using pip:

.. code-block:: bash

   pip install ocpy

From Source (Development)
^^^^^^^^^^^^^^^^^^^^^^^^^

For the latest development version or to contribute:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/ocean-colour/ocpy.git
   cd ocpy

   # Install in editable mode
   pip install -e .

   # Or with all optional dependencies
   pip install -e ".[all]"

Using Conda
^^^^^^^^^^^

You can create a conda environment with all dependencies:

.. code-block:: bash

   # Create a new environment
   conda create -n ocpy python=3.10

   # Activate the environment
   conda activate ocpy

   # Install dependencies
   conda install numpy scipy pandas xarray matplotlib seaborn scikit-learn
   conda install -c conda-forge cartopy geopandas healpy netcdf4

   # Install ocpy
   pip install -e /path/to/ocpy

Verifying Installation
----------------------

To verify that ocpy is installed correctly:

.. code-block:: python

   import oceancolor

   # Check version
   print(f"ocpy version: {oceancolor.__version__}")

   # Test basic functionality
   from oceancolor.water.absorption import a_water
   import numpy as np

   wavelengths = np.array([443, 490, 555, 670])
   a_w = a_water(wavelengths)
   print(f"Water absorption at {wavelengths}: {a_w}")

You should see output like:

.. code-block:: text

   ocpy version: 0.1.dev0
   Water absorption at [443 490 555 670]: [0.0071 0.0148 0.0565 0.439 ]

Running Tests
-------------

To run the test suite:

.. code-block:: bash

   # Run all tests
   pytest ocpy/tests/

   # Run with verbose output
   pytest -v ocpy/tests/

   # Run a specific test file
   pytest ocpy/tests/test_water.py

   # Run a specific test
   pytest ocpy/tests/test_ls2.py::test_ls2_run

External Data Setup
-------------------

Some modules require external datasets that are not distributed with the package.

Hydrolight Data (Loisel+2023)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``hydrolight`` module requires the Loisel+2023 Hydrolight simulation dataset:

1. Download from Dryad: https://doi.org/10.6076/D1630T
2. Set the ``OS_COLOR`` environment variable to the download location:

.. code-block:: bash

   # In your shell configuration (.bashrc, .zshrc, etc.)
   export OS_COLOR="/path/to/loisel2023/data"

3. Verify the setup:

.. code-block:: python

   from oceancolor.hydrolight.loisel23 import load_ds

   # Load dataset with Raman scattering, 30-degree solar zenith
   ds = load_ds(X=2, Y=30)
   print(ds)

Tara Oceans Data
^^^^^^^^^^^^^^^^

The Tara Oceans expedition data requires downloading parquet files:

1. See ``ocpy/data/Tara/README.md`` for download instructions
2. Place files in the ``ocpy/data/Tara/`` directory

Troubleshooting
---------------

Import Errors
^^^^^^^^^^^^^

If you encounter import errors, ensure all dependencies are installed:

.. code-block:: bash

   pip install seaborn pyarrow healpy xarray h5netcdf scikit-learn cartopy netcdf4 geopandas

Cartopy Installation Issues
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Cartopy can be difficult to install via pip on some systems. Try conda:

.. code-block:: bash

   conda install -c conda-forge cartopy

HEALPix/healpy Issues
^^^^^^^^^^^^^^^^^^^^^

On some systems, healpy requires additional system libraries:

.. code-block:: bash

   # Ubuntu/Debian
   sudo apt-get install libcfitsio-dev

   # macOS with Homebrew
   brew install cfitsio

NetCDF Issues
^^^^^^^^^^^^^

If you encounter netCDF-related errors:

.. code-block:: bash

   # Ubuntu/Debian
   sudo apt-get install libnetcdf-dev libhdf5-dev

   # macOS with Homebrew
   brew install netcdf hdf5

Getting Help
------------

If you encounter issues:

1. Check the `GitHub Issues <https://github.com/ocean-colour/ocpy/issues>`_ for known problems
2. Open a new issue with:

   * Your Python version (``python --version``)
   * Your operating system
   * The full error traceback
   * A minimal code example that reproduces the issue
