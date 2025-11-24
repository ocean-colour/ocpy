# PACE Module

This directory contains utilities for working with PACE (Plankton, Aerosol, Cloud, ocean Ecosystem) mission data.

## Modules

### io.py

Functions for loading PACE OCI (Ocean Color Instrument) L2 data from netCDF files.

**Key Functions:**
- `load_oci_l2(fn, full_flag)`: Load OCI L2 reflectance data (Rrs, Rrs_unc)
- `load_iop_l2(fn)`: Load IOP (Inherent Optical Properties) L2 data

**Data Structure:**
PACE data is returned as xarray Datasets with dimensions:
- `x`, `y`: Spatial dimensions (along-track and cross-track)
- `wl`: Wavelength dimension (typically 184 wavelengths from 339-719 nm)

### correlated_error.py

Comprehensive module for analyzing correlated errors in PACE Rrs measurements.

**Purpose:**
PACE provides uncertainty estimates (Rrs_unc) for each wavelength, but these uncertainties are correlated across wavelengths. This module calculates the correlation structure and generates full error covariance matrices for proper uncertainty propagation.

**Key Functions:**

1. **calc_global_correlation(xds, flags, ...)**
   - Calculates correlation matrix using all valid pixels in the granule
   - Fast and provides consistent correlation estimates
   - Recommended for most applications
   - Returns: correlation_matrix (n_wl × n_wl), wavelengths

2. **calc_local_correlation(xds, window_size, ...)**
   - Calculates pixel-specific correlations using spatial neighborhoods
   - Can capture spatial variability in correlations
   - More computationally intensive
   - Returns: correlation_matrices (x × y × n_wl × n_wl), wavelengths

3. **calc_error_covariance_matrices(xds, correlation_matrix, ...)**
   - Generates error covariance matrices for each pixel
   - Formula: Cov[i,j] = Corr[i,j] × σ[i] × σ[j]
   - Returns: covariance_matrices (x × y × n_wl × n_wl), wavelengths

4. **process_granule(filename, method, ...)**
   - Complete pipeline: load → correlate → generate covariances
   - Convenience function for typical workflows
   - Optionally saves results to compressed npz file
   - Returns: dictionary with xds, flags, correlations, covariances

**Utility Functions:**
- `extract_pixel_covariance(cov_matrices, x, y)`: Extract covariance for specific pixel
- `get_correlation_summary(corr_matrix)`: Calculate summary statistics

## Usage Examples

### Basic Usage: Global Correlation

```python
from ocpy.pace import correlated_error as ce

# Process entire granule with global correlation
results = ce.process_granule(
    'PACE_OCI.20240416T093158.L2.OC_AOP.nc',
    method='global',
    save_results=True
)

# Access results
correlation_matrix = results['correlation_matrix']  # (n_wl, n_wl)
covariance_matrices = results['covariance_matrices']  # (x, y, n_wl, n_wl)
wavelengths = results['wavelengths']

# Extract covariance for a specific pixel
pixel_cov = ce.extract_pixel_covariance(covariance_matrices, x=500, y=600)
```

### Advanced: Local Correlation

```python
# Use spatial neighborhoods for correlation
results = ce.process_granule(
    'PACE_file.nc',
    method='local',
    window_size=20  # 20×20 pixel window
)

# Each pixel has its own correlation matrix
local_corr = results['correlation_matrices']  # (x, y, n_wl, n_wl)
```

### Wavelength Subset

```python
# Process only specific wavelengths (faster)
subset_indices = np.arange(0, 184, 5)  # Every 5th wavelength

results = ce.process_granule(
    'PACE_file.nc',
    method='global',
    wavelength_subset=subset_indices
)
```

### Manual Workflow

```python
from ocpy.pace import io as pace_io
from ocpy.pace import correlated_error as ce

# 1. Load data
xds, flags = pace_io.load_oci_l2('PACE_file.nc', full_flag=False)

# 2. Calculate correlation
corr_matrix, wavelengths = ce.calc_global_correlation(
    xds,
    flags=flags,
    flag_value=0
)

# 3. Generate covariance matrices
cov_matrices, _ = ce.calc_error_covariance_matrices(
    xds,
    correlation_matrix=corr_matrix
)

# 4. Use in downstream analysis
pixel_cov = cov_matrices[100, 200, :, :]  # Pixel (100, 200)
# Now propagate uncertainties using this covariance matrix
```

## Applications

### Uncertainty Propagation

When calculating derived quantities from Rrs (e.g., chlorophyll, IOPs), proper uncertainty propagation requires the full error covariance matrix:

```python
# Example: Uncertainty in sum of Rrs values
pixel_cov = covariance_matrices[i, j, :, :]

# Variance of sum: Var(ΣX) = Σ Cov[i,j]
total_variance = np.sum(pixel_cov)
total_uncertainty = np.sqrt(total_variance)
```

### Band Ratios

For band ratio algorithms (e.g., OC4), correlations are critical:

```python
# Uncertainty in ratio R = Rrs[i] / Rrs[j]
# Using first-order approximation:
# Var(R) ≈ (∂R/∂Rrs_i)² Var[i] + (∂R/∂Rrs_j)² Var[j] + 2(∂R/∂Rrs_i)(∂R/∂Rrs_j) Cov[i,j]

i, j = 10, 50  # Wavelength indices
cov = pixel_cov[i, j]
var_i = pixel_cov[i, i]
var_j = pixel_cov[j, j]

# Derivatives
dR_di = 1 / Rrs[j]
dR_dj = -Rrs[i] / Rrs[j]**2

var_ratio = dR_di**2 * var_i + dR_dj**2 * var_j + 2 * dR_di * dR_dj * cov
```

### Inversion Algorithms

For algorithms like LS2 that use multiple wavelengths, the error covariance matrix is essential for proper uncertainty estimation of retrieved IOPs.

## Testing

Tests are located in `ocpy/tests/test_correlated_error.py`. Run with:

```bash
pytest ocpy/tests/test_correlated_error.py -v
```

Tests use synthetic data to verify:
- Correlation matrix properties (symmetric, diagonal = 1, values in [-1, 1])
- Covariance matrix properties (symmetric, positive semi-definite)
- Error handling and edge cases

## Example Scripts

See `examples/pace_correlated_error_example.py` for comprehensive examples including:
- Global and local correlation methods
- Pixel-level analysis
- Visualization of correlation structure
- Spatial variability in uncertainties

## Performance Notes

- **Global correlation**: Fast (~seconds for typical granule), recommended for most uses
- **Local correlation**: Slow (~minutes), use when spatial variability is important
- **Memory**: Covariance matrices can be large (x × y × n_wl × n_wl). For typical granule (1700 × 1300 × 184), this is ~300 GB if stored in float64. Consider:
  - Using wavelength subsets
  - Processing in chunks
  - Storing in float32
  - Computing on-the-fly rather than storing all matrices

## References

- PACE mission: https://pace.gsfc.nasa.gov/
- Data access: https://oceandata.sci.gsfc.nasa.gov/
- OCI data format: PACE OCI L2 Data Format Specification
