# PACE Correlated Error Analysis - Quick Start Guide

## What is this?

PACE provides uncertainty estimates (`Rrs_unc`) for each wavelength, but these uncertainties are **correlated** across wavelengths. This module calculates the correlation structure and generates full error **covariance matrices** for proper uncertainty propagation in your analysis.

## Why do I need this?

If you're doing any of the following, you need error covariance matrices:

1. **Calculating derived products** (chlorophyll, IOPs, etc.) and want proper uncertainties
2. **Using band ratios** (correlations significantly affect ratio uncertainties)
3. **Running inversions** (LS2, QAA, etc.) with proper error propagation
4. **Combining wavelengths** in any mathematical operation

Simply using `Rrs_unc` alone will **underestimate** uncertainties if wavelengths are positively correlated (typical case) or **overestimate** if negatively correlated.

## Quick Start (3 lines of code)

```python
from ocpy.pace import correlated_error as ce

# Process your PACE file (this does everything)
results = ce.process_granule('your_PACE_file.nc', method='global')

# Get error covariance for pixel (x, y)
pixel_cov = results['covariance_matrices'][100, 200, :, :]
```

That's it! You now have the full error covariance matrix for each pixel.

## What's in the results?

```python
results = {
    'xds': xarray Dataset with Rrs and Rrs_unc,
    'flags': Quality flags array,
    'wavelengths': Array of wavelengths used,
    'correlation_matrix': (n_wl, n_wl) correlation between wavelengths,
    'covariance_matrices': (x, y, n_wl, n_wl) error covariance for each pixel
}
```

## How to use the covariance matrix

### Example 1: Uncertainty in a sum

```python
# Get covariance for pixel (i, j)
cov = results['covariance_matrices'][i, j, :, :]

# Uncertainty in sum of all wavelengths
# Var(ΣX) = Σ_ij Cov[i,j]
total_var = np.sum(cov)
total_unc = np.sqrt(total_var)
```

### Example 2: Uncertainty in a band ratio

```python
import numpy as np

# Get Rrs and covariance for pixel (i, j)
Rrs = results['xds']['Rrs'].values[i, j, :]
cov = results['covariance_matrices'][i, j, :, :]

# Calculate ratio: R = Rrs[wl1] / Rrs[wl2]
wl1, wl2 = 10, 50  # wavelength indices
ratio = Rrs[wl1] / Rrs[wl2]

# Propagate uncertainty using first-order Taylor expansion
# Var(R) ≈ (∂R/∂x)² Var(x) + (∂R/∂y)² Var(y) + 2(∂R/∂x)(∂R/∂y) Cov(x,y)
dR_dwl1 = 1 / Rrs[wl2]
dR_dwl2 = -Rrs[wl1] / Rrs[wl2]**2

var_ratio = (dR_dwl1**2 * cov[wl1, wl1] +
             dR_dwl2**2 * cov[wl2, wl2] +
             2 * dR_dwl1 * dR_dwl2 * cov[wl1, wl2])

ratio_unc = np.sqrt(var_ratio)
```

### Example 3: Multi-wavelength inversion

```python
# For algorithms using multiple wavelengths (e.g., LS2)
# Use the full covariance matrix in your inversion

def my_inversion(Rrs, Rrs_cov):
    """
    Your inversion algorithm.

    Parameters:
        Rrs: (n_wl,) array of reflectances
        Rrs_cov: (n_wl, n_wl) error covariance matrix

    Returns:
        result: Retrieved parameter
        result_unc: Uncertainty on retrieved parameter
    """
    # ... your inversion code ...

    # Propagate uncertainties using Rrs_cov
    # This typically involves Jacobian calculation
    J = calculate_jacobian(Rrs)  # Your function
    result_cov = J @ Rrs_cov @ J.T
    result_unc = np.sqrt(np.diag(result_cov))

    return result, result_unc
```

## Common Options

### Only process specific wavelengths (faster)

```python
# Process every 10th wavelength
subset = np.arange(0, 184, 10)
results = ce.process_granule('file.nc', wavelength_subset=subset)
```

### Use spatial neighborhoods (captures local variability)

```python
# Each pixel gets its own correlation matrix from nearby pixels
results = ce.process_granule('file.nc', method='local', window_size=20)
```

### Save results for later

```python
# Saves to compressed .npz file
results = ce.process_granule('file.nc', save_results=True)

# Load later
data = np.load('file_covariance.npz', allow_pickle=True)
cov_matrices = data['covariance_matrices']
```

## Typical Correlation Structure

PACE Rrs correlations typically show:
- **High positive correlation** between nearby wavelengths (0.8-0.95)
- **Moderate correlation** between distant wavelengths (0.3-0.7)
- **Correlation decreases** with wavelength separation
- **Some negative correlations** may appear at specific wavelength pairs

You can check this:
```python
corr = results['correlation_matrix']
summary = ce.get_correlation_summary(corr)
print(f"Mean correlation: {summary['mean']:.3f}")
print(f"Range: [{summary['min']:.3f}, {summary['max']:.3f}]")
```

## Need More Details?

- Full documentation: `ocpy/pace/README.md`
- Example script: `examples/pace_correlated_error_example.py`
- Tests: `ocpy/tests/test_correlated_error.py`

## Citation

If you use this module in your research, please cite the ocpy package and acknowledge the PACE mission.
