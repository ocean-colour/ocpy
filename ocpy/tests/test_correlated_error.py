"""
Tests for correlated error analysis module.

Note: These tests use synthetic data. For testing with real PACE data,
see examples/pace_error_example.py
"""

import numpy as np
import xarray as xr
import pytest

from ocpy.pace import correlated_error as ce


def create_synthetic_pace_data(n_x=50, n_y=50, n_wl=20, add_noise=True):
    """
    Create synthetic PACE-like data for testing.

    Parameters:
        n_x (int): Number of pixels in x dimension
        n_y (int): Number of pixels in y dimension
        n_wl (int): Number of wavelengths
        add_noise (bool): Whether to add random noise

    Returns:
        xds (xr.Dataset): Synthetic dataset
        flags (np.ndarray): Synthetic flags
    """
    # Create wavelengths
    wavelengths = np.linspace(400, 700, n_wl).astype(int)

    # Create synthetic Rrs data with spectral structure
    # Use a combination of exponential decay and Gaussian features
    base_spectrum = np.exp(-0.005 * (wavelengths - 400)) + \
                    0.5 * np.exp(-0.01 * (wavelengths - 550)**2)

    # Add spatial variability
    x_coords = np.arange(n_x)
    y_coords = np.arange(n_y)
    spatial_var = np.outer(np.sin(x_coords / 10), np.cos(y_coords / 10))

    # Combine spectral and spatial components
    Rrs = np.zeros((n_x, n_y, n_wl))
    for i in range(n_wl):
        Rrs[:, :, i] = base_spectrum[i] * (1 + 0.3 * spatial_var)

    # Add noise if requested
    if add_noise:
        Rrs += np.random.normal(0, 0.0001, Rrs.shape)

    # Create uncertainties (proportional to signal)
    Rrs_unc = 0.05 * Rrs + 0.00005

    # Create flags (random flagging of ~10% of pixels)
    flags = np.zeros((n_x, n_y), dtype=int)
    n_flagged = int(0.1 * n_x * n_y)
    flag_indices = np.random.choice(n_x * n_y, n_flagged, replace=False)
    flags.flat[flag_indices] = 1

    # Create coordinates
    lats = np.linspace(-30, -20, n_x * n_y).reshape(n_x, n_y)
    lons = np.linspace(40, 50, n_x * n_y).reshape(n_x, n_y)

    # Create xarray Dataset
    xds = xr.Dataset(
        {
            'Rrs': (['x', 'y', 'wl'], Rrs),
            'Rrs_unc': (['x', 'y', 'wl'], Rrs_unc)
        },
        coords={
            'latitude': (['x', 'y'], lats),
            'longitude': (['x', 'y'], lons),
            'wavelength': ('wl', wavelengths)
        }
    )

    return xds, flags


def test_global_correlation():
    """Test global correlation calculation."""
    xds, flags = create_synthetic_pace_data(n_x=30, n_y=30, n_wl=15)

    # Calculate global correlation
    corr_matrix, wavelengths = ce.calc_global_correlation(
        xds, flags=flags, flag_value=0
    )

    # Check shape
    assert corr_matrix.shape == (15, 15)
    assert len(wavelengths) == 15

    # Check properties of correlation matrix
    # Diagonal should be 1 (or very close)
    assert np.allclose(np.diag(corr_matrix), 1.0, atol=1e-6)

    # Should be symmetric
    assert np.allclose(corr_matrix, corr_matrix.T, atol=1e-6)

    # Values should be in [-1, 1]
    assert np.all(corr_matrix >= -1.0) and np.all(corr_matrix <= 1.0)


def test_global_correlation_with_subset():
    """Test global correlation with wavelength subset."""
    xds, flags = create_synthetic_pace_data(n_x=30, n_y=30, n_wl=20)

    # Use subset of wavelengths
    subset_indices = np.array([0, 5, 10, 15, 19])

    corr_matrix, wavelengths = ce.calc_global_correlation(
        xds, flags=flags, flag_value=0, wavelength_subset=subset_indices
    )

    # Check shape matches subset
    assert corr_matrix.shape == (5, 5)
    assert len(wavelengths) == 5


def test_local_correlation():
    """Test local correlation calculation."""
    xds, flags = create_synthetic_pace_data(n_x=30, n_y=30, n_wl=10)

    # Calculate local correlation with small window
    corr_matrices, wavelengths = ce.calc_local_correlation(
        xds, window_size=5, flags=flags, flag_value=0, min_samples=10
    )

    # Check shape
    assert corr_matrices.shape == (30, 30, 10, 10)
    assert len(wavelengths) == 10

    # At least some correlation matrices should be valid
    n_valid = np.sum(np.all(np.isfinite(corr_matrices), axis=(2, 3)))
    assert n_valid > 0


def test_error_covariance_global():
    """Test error covariance matrix generation with global correlation."""
    xds, flags = create_synthetic_pace_data(n_x=20, n_y=20, n_wl=10)

    # Calculate global correlation
    corr_matrix, _ = ce.calc_global_correlation(
        xds, flags=flags, flag_value=0
    )

    # Generate covariance matrices
    cov_matrices, wavelengths = ce.calc_error_covariance_matrices(
        xds, correlation_matrix=corr_matrix
    )

    # Check shape
    assert cov_matrices.shape == (20, 20, 10, 10)

    # Check that at least some are valid
    n_valid = np.sum(np.all(np.isfinite(cov_matrices), axis=(2, 3)))
    assert n_valid > 0

    # For a valid pixel, check covariance matrix properties
    # Find a valid pixel
    for i in range(20):
        for j in range(20):
            if np.all(np.isfinite(cov_matrices[i, j, :, :])):
                cov = cov_matrices[i, j, :, :]

                # Should be symmetric
                assert np.allclose(cov, cov.T, atol=1e-6)

                # Should be positive semi-definite (all eigenvalues >= 0)
                eigenvalues = np.linalg.eigvals(cov)
                assert np.all(eigenvalues >= -1e-10)  # Allow small numerical errors

                break
        else:
            continue
        break


def test_extract_pixel_covariance():
    """Test extracting covariance for specific pixel."""
    xds, flags = create_synthetic_pace_data(n_x=20, n_y=20, n_wl=10)

    corr_matrix, _ = ce.calc_global_correlation(xds, flags=flags)
    cov_matrices, _ = ce.calc_error_covariance_matrices(
        xds, correlation_matrix=corr_matrix
    )

    # Extract covariance for a specific pixel
    pixel_cov = ce.extract_pixel_covariance(cov_matrices, 10, 10, check_valid=False)

    # Check shape
    assert pixel_cov.shape == (10, 10)


def test_extract_pixel_covariance_invalid():
    """Test that extracting invalid pixel raises error."""
    cov_matrices = np.random.randn(20, 20, 10, 10)

    # Out of bounds should raise error
    with pytest.raises(ValueError):
        ce.extract_pixel_covariance(cov_matrices, 25, 10)

    with pytest.raises(ValueError):
        ce.extract_pixel_covariance(cov_matrices, 10, 25)


def test_correlation_summary():
    """Test correlation summary statistics."""
    # Create a simple correlation matrix
    n = 5
    corr_matrix = np.eye(n)
    # Add some off-diagonal correlations
    for i in range(n-1):
        corr_matrix[i, i+1] = 0.8
        corr_matrix[i+1, i] = 0.8

    summary = ce.get_correlation_summary(corr_matrix)

    # Check that summary contains expected keys
    assert 'mean' in summary
    assert 'std' in summary
    assert 'min' in summary
    assert 'max' in summary
    assert 'n_negative' in summary

    # Check values are reasonable
    assert summary['min'] >= -1.0
    assert summary['max'] <= 1.0
    assert summary['n_negative'] >= 0


def test_insufficient_samples():
    """Test that insufficient samples raises error."""
    # Create very small dataset with lots of NaNs
    xds, flags = create_synthetic_pace_data(n_x=5, n_y=5, n_wl=10)

    # Flag most pixels
    flags[:] = 1
    flags[0, 0] = 0  # Only one valid pixel

    # Should raise error due to insufficient samples
    with pytest.raises(ValueError, match="Insufficient valid samples"):
        ce.calc_global_correlation(xds, flags=flags, flag_value=0, min_samples=100)


def test_no_correlation_matrix_error():
    """Test that error is raised when no correlation matrix provided."""
    xds, _ = create_synthetic_pace_data(n_x=10, n_y=10, n_wl=5)

    # Should raise error
    with pytest.raises(ValueError, match="Either correlation_matrix or correlation_matrices"):
        ce.calc_error_covariance_matrices(xds)
