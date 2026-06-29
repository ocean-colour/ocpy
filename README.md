# ocpy

Python software repository to support ocean color analyses --
*here, there, and everywhere.*

[![Tests](https://github.com/ocean-colour/ocpy/actions/workflows/tests.yml/badge.svg)](https://github.com/ocean-colour/ocpy/actions/workflows/tests.yml)
[![Documentation Status](https://readthedocs.org/projects/ocpy/badge/?version=latest)](https://ocpy.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/644861243.svg)](https://doi.org/10.5281/zenodo.17088614)

## Overview

`ocpy` provides tools for ocean color remote sensing and bio-optical
analysis, including:

- **Water optical properties** -- absorption and scattering of pure
  seawater (GSFC / IOCCG 2018 data, Zhang et al. 2009 scattering).
- **IOP inversions** -- the LS2 model (Loisel et al. 2018) and ZLee
  methods to derive absorption / backscattering from `Rrs`.
- **Chlorophyll-a algorithms** -- OC2 / OC4 band-ratio retrievals.
- **Phytoplankton absorption** -- Bricaud (1998) parameterizations.
- **Satellite data processing** -- PACE, MODIS, and SeaWiFS utilities
  (wavelengths, noise models, granule readers).
- **Field / cruise data** -- Tara Oceans ingest and the Hyper-a
  integrating-cavity absorption meter workflow.
- **Hydrolight outputs** -- loaders for the Loisel et al. (2023)
  synthetic dataset.

## Installation

`ocpy` is distributed on PyPI as **`ocpy-ocean`** (the bare name `ocpy`
is taken by an unrelated project); the import package is still `ocpy`:

```bash
pip install ocpy-ocean
```

For a development / editable install from a clone:

```bash
pip install -e .          # runtime dependencies
pip install -e .[dev]     # + pytest for the test suite
pip install -e .[docs]    # + Sphinx for the documentation build
```

Python 3.10+ is required.

## Command-line tools

Installing the package exposes two console scripts:

```bash
ocpy_view --help        # multi-panel view of ocean color granules
ocpy_plot_rrs --help    # plot a single PACE Rrs spectrum
```

## Testing

```bash
pytest ocpy/tests/
```

Tests that depend on large external datasets skip automatically when the
data are not present.

## External data

Some modules require datasets configured via environment variables:

- `OS_COLOR` -- path to the Loisel et al. (2023) Hydrolight datasets
  (Dryad: [doi:10.6076/D1630T](https://doi.org/10.6076/D1630T)).
- Tara parquet tables -- download per `ocpy/data/Tara/README.md`.

## Documentation

Full documentation is hosted at
[ocpy.readthedocs.io](https://ocpy.readthedocs.io).

## Citation

If you use `ocpy` in your research, please cite it via its Zenodo DOI:
[10.5281/zenodo.17088614](https://doi.org/10.5281/zenodo.17088614).
