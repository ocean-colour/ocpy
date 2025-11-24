# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`ocpy` (Ocean Color Python) is a Python package for ocean color analysis, focused on remote sensing reflectance (Rrs), inherent optical properties (IOPs), chlorophyll-a estimation, and satellite data processing. The package implements various oceanographic algorithms including the LS2 inversion model (Loisel et al. 2018) for estimating optical absorption and backscattering coefficients.

## Package Structure

The codebase is organized into modules with distinct responsibilities:

- **ls2/**: LS2 inversion model implementation
  - `ls2_main.py`: Core LS2 algorithm for calculating absorption (a), backscattering (bb), and related coefficients from Rrs
  - `kd_nn.py`: Neural network for estimating Kd (attenuation coefficient)
  - `io.py`: Loading look-up tables (LUTs) from package data

- **satellites/**: Satellite data processing and error models
  - `modis.py`: MODIS-Aqua specific functions, matchups, and error calculations
  - `pace.py`: PACE mission functions including noise vector generation
  - `seawifs.py`, `sbg.py`: Other satellite-specific utilities

- **chl/**: Chlorophyll-a estimation algorithms
  - `band_ratios.py`: OC2, OC4 algorithms for estimating chlorophyll from Rrs

- **water/**: Pure seawater optical properties
  - `absorption.py`: Water absorption coefficients (GSFC RSR, IOCCG 2018 data)
  - `scattering.py`: Water scattering coefficients

- **iop/**: Inherent optical properties (IOPs)
  - `cdom.py`: CDOM (colored dissolved organic matter) absorption modeling using exponential and power law functions
  - `cross.py`, `zlee.py`: Additional IOP models

- **hydrolight/**: Hydrolight simulation data interfaces
  - `loisel23.py`: Interface to Loisel+2023 Hydrolight outputs (requires OS_COLOR environment variable)

- **pace/**: PACE mission specific utilities
  - `io.py`: PACE data I/O functions for loading L2 OCI and IOP products
  - `correlated_error.py`: Correlation and error covariance analysis for Rrs data

- **insitu/**: In-situ measurement handling
  - `gloria.py`: GLORIA dataset processing

- **tara/**, **ph/**, **polarize/**: Domain-specific modules for Tara Oceans data, phytoplankton, and polarization

- **utils/**: Shared utilities
  - `coords.py`: Coordinate conversion functions (DMS to decimal)
  - `cat_utils.py`, `fig_utils.py`: Catalog and figure utilities

- **tests/**: Unit tests for all major modules

## Development Commands

### Installation

The package is installed in editable mode:
```bash
pip install -e .
```

Required dependencies include: numpy, scipy, pandas, xarray, seaborn, healpy, cartopy, geopandas, and others listed in setup.py.

### Running Tests

Run all tests:
```bash
pytest ocpy/tests/
```

Run a single test file:
```bash
pytest ocpy/tests/test_ls2.py
```

Run a specific test function:
```bash
pytest ocpy/tests/test_ls2.py::test_ls2_run
```

**Important**: Some tests import from `oceancolor` instead of `ocpy` due to legacy naming. The package must be installed for tests to run properly.

### Linting and Code Quality

No specific linters are configured in the repository. Standard Python conventions apply.

## Key Architecture Notes

### LS2 Inversion Model

The LS2 model (ocpy/ls2/ls2_main.py) is a central algorithm that:
1. Takes solar zenith angle (sza), wavelength, Rrs, Kd, and scattering coefficients as inputs
2. Uses look-up tables (LUTs) to interpolate absorption and backscattering coefficients
3. Optionally applies Raman scattering correction (Flag_Raman parameter)
4. Returns: a (absorption), anw (non-water absorption), bb (backscattering), bbp (particulate backscattering), kappa (Raman correction factor)

The model relies heavily on pre-computed LUTs stored in `ocpy/data/LS2/` which are loaded via `ls2_io.load_LUT()`.

### Package Data Access

Data files are accessed using `pkg_resources.resource_filename()` or `importlib.resources.files()`:
```python
filename = resource_filename('oceancolor', os.path.join('data', 'LS2', 'LS2_LUT.npz'))
```

**Note**: Some code uses 'oceancolor' as the package name (legacy), while the actual package is 'ocpy'. Be consistent with 'ocpy' in new code.

### Satellite Error Modeling

Satellite modules (modis.py, pace.py) provide:
- Wavelength-specific measurement uncertainties
- Error vectors for noise injection in simulations
- Matchup datasets for validation (e.g., MODIS_matchups_rrs.csv)

### Water Optical Properties

The package provides multiple sources for water absorption coefficients:
- GSFC RSR tables (default)
- IOCCG 2018 data (more recent, provided by R. Reynolds at Scripps)

Access via `ocpy.water.absorption.a_water(wavelengths, data='GSFC')`.

### PACE Correlated Error Analysis

The `ocpy.pace.correlated_error` module provides functionality for analyzing correlated errors in PACE Rrs data:

**Key Functions:**
- `calc_global_correlation()`: Calculate correlation matrix from entire granule (fast, recommended for most uses)
- `calc_local_correlation()`: Calculate pixel-specific correlations using spatial neighborhoods
- `calc_error_covariance_matrices()`: Generate error covariance matrices using Rrs_unc and correlations
- `process_granule()`: Convenience function for complete processing pipeline

**Workflow:**
1. Load PACE L2 data using `pace.io.load_oci_l2()`
2. Calculate correlation between Rrs wavelengths (global or local method)
3. Generate pixel-by-pixel error covariance matrices: `Cov[i,j] = Corr[i,j] * sigma[i] * sigma[j]`

**Usage:**
```python
from ocpy.pace import correlated_error as ce

# Full pipeline with global correlation (recommended)
results = ce.process_granule('PACE_file.nc', method='global')
cov_matrices = results['covariance_matrices']  # Shape: (x, y, n_wl, n_wl)

# Extract covariance for specific pixel
pixel_cov = ce.extract_pixel_covariance(cov_matrices, x=100, y=200)
```

See `examples/pace_correlated_error_example.py` for detailed usage examples.

### Environment Variables

- `OS_COLOR`: Required for accessing Hydrolight simulation data (ocpy/hydrolight/loisel23.py). If not set, falls back to current directory with a warning.
- `OS_COLOR`: Also used in some example scripts for PACE data paths

## Data Directory Structure

Package data is stored in `ocpy/data/`:
- `LS2/`: LS2 look-up tables (LUT.npz) and Kd neural network weights
- `satellites/`: Satellite error tables (PACE_error.csv, MODIS_matchups_rrs.csv)
- `water/`: Water absorption/scattering coefficients
- `phytoplankton/`, `Tara/`, `COASTLOOC/`, `polarization/`, `Rrs/`: Domain-specific datasets

## Common Patterns

### Wavelength Arrays
Wavelengths are typically in nanometers (nm), stored as numpy arrays. PACE uses 5 nm sampling by default (400-700 nm).

### Spectral Coefficients
Optical coefficients are in units of m^-1 (per meter).

### Remote Sensing Reflectance (Rrs)
Units are sr^-1 (per steradian). This is the primary input for most algorithms.

### Interpolation
Heavy use of scipy interpolation functions (interp1d, RegularGridInterpolator) for wavelength-dependent properties and LUT lookups.

## Testing Philosophy

Tests are comprehensive for core algorithms (especially LS2). Test data files are stored in `ocpy/tests/files/`. Tests compare outputs against reference Excel files (e.g., LS2_test_run.xls) using numpy.allclose() with appropriate tolerances.

## Known Issues

- Mixed package naming: Some imports use 'oceancolor', others use 'ocpy'
- Tests may fail if package is not properly installed in the environment
- Hydrolight functionality requires external data downloaded from Dryad
