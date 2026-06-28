# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ocpy is a Python package for ocean color analysis. It provides tools for:
- Water optical properties (absorption, scattering)
- Inherent Optical Properties (IOP) inversions (LS2 model, ZLee methods)
- Chlorophyll-a algorithms (OC2, OC4 band ratios)
- Phytoplankton absorption
- Satellite data processing (PACE, MODIS, SeaWiFS)
- Tara Oceans expedition data processing
- Hydrolight simulation outputs (Loisel+2023)

## Build and Install

```bash
pip install -e .          # runtime deps
pip install -e .[dev]     # + pytest
```

Packaging metadata and dependencies now live in `pyproject.toml` (the
single source of truth); `setup.py` is a thin shim for legacy/editable
installs. On PyPI the distribution is named `ocpy-ocean` (the import
package stays `ocpy`). Runtime deps: numpy, scipy, pandas, matplotlib,
seaborn, xarray, scikit-learn, netcdf4, h5netcdf, pyarrow, geopandas,
shapely, bokeh, ipython.

## Running Tests

```bash
# Run all tests
pytest ocpy/tests/

# Run a single test file
pytest ocpy/tests/test_water.py

# Run a specific test
pytest ocpy/tests/test_ls2.py::test_ls2_run
```

Note: Tests import from `oceancolor` namespace (e.g., `from oceancolor.water.scattering import PMH`), not `ocpy`.

## Code Architecture

### Module Structure

- **`water/`**: Pure seawater optical properties
  - `absorption.py`: Water absorption coefficients (GSFC, IOCCG 2018 datasets)
  - `scattering.py`: Water scattering (Zhang et al. 2009 model, refractive index calculations)

- **`ls2/`**: LS2 inversion model (Loisel et al. 2018) for deriving absorption and backscattering from Rrs
  - `ls2_main.py`: Main LS2 algorithm with look-up table interpolation
  - `io.py`: LUT loading functions
  - `kd_nn.py`: Neural network for Kd estimation

- **`chl/`**: Chlorophyll algorithms
  - `band_ratios.py`: OC2, OC4 algorithms

- **`iop/`**: Inherent Optical Properties
  - `zlee.py`: ZLee IOP methods
  - `cdom.py`: CDOM absorption

- **`ph/`**: Phytoplankton
  - `absorption.py`: Phytoplankton absorption (Bricaud 1998)
  - `pigments.py`: Pigment analysis

- **`satellites/`**: Satellite-specific utilities
  - `pace.py`: PACE wavelengths, noise model
  - `modis.py`, `seawifs.py`: Sensor-specific functions

- **`tara/`**: Tara Oceans expedition data
  - `io.py`, `ingest.py`: Data loading
  - `analysis.py`, `spectra.py`: Analysis routines

- **`hydrolight/`**: Radiative transfer simulation data
  - `loisel23.py`: Load Loisel+2023 Hydrolight datasets (requires OS_COLOR env var)

- **`hyper_a/`**: Hyper-a integrating cavity absorption meter processing (Sequoia Scientific)
  - `io.py`: Binary file reader, calibration loading
  - `lib.py`: Core processing functions (IOCCG water absorption, transmission, absorption computation)
  - `process.py`: Main processing workflow (`process()`, `rho_from_nd_spot()`)
  - `matlab/`: Original MATLAB reference code

### Data Files

Reference data is stored in `ocpy/data/` and accessed via `importlib.resources`:
- `water/`: Water absorption/scattering coefficients
- `LS2/`: Look-up tables for LS2 model
- `phytoplankton/`: Absorption spectra (Bricaud, Kramer, Moore)
- `satellites/`: PACE error model, matchup datasets
- `Tara/`: Tara expedition data

### External Data

Some modules require external datasets configured via environment variables:
- `OS_COLOR`: Path to Loisel+2023 Hydrolight datasets (download from Dryad: doi:10.6076/D1630T)
- Tara parquet files: Download separately per `ocpy/data/Tara/README.md`

## Conventions

- Wavelengths are in nanometers (nm)
- Absorption/scattering coefficients are in m^-1
- Rrs (remote-sensing reflectance) is in sr^-1
- Functions typically accept `wave` (wavelength array) and `Rrs` (reflectance array) as inputs
- Uses xarray for Hydrolight netCDF data, pandas for tabular data
