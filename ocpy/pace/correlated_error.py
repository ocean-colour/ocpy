"""
Correlated error analysis for PACE OCI data.

This module provides functionality to calculate correlation matrices between
Rrs wavelengths and generate pixel-by-pixel error covariance matrices using
the Rrs uncertainties provided by PACE.

Functions:
    - calc_global_correlation: Calculate correlation from entire granule
    - calc_local_correlation: Calculate correlation from spatial neighborhoods
    - calc_error_covariance_matrices: Generate error covariance matrix for each pixel
    - process_granule: Full pipeline for processing PACE granules

Author: ocpy team
"""

import numpy as np
import xarray as xr
from typing import Tuple, Optional, Union
import warnings

from IPython import embed


def calc_global_correlation(
    xds: xr.Dataset,
    flags: Optional[np.ndarray] = None,
    flag_value: int = 0,
    wavelength_subset: Optional[np.ndarray] = None,
    min_samples: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate global correlation matrix between Rrs wavelengths using all valid pixels.

    This function computes the Pearson correlation coefficient between all pairs of
    wavelengths using valid (unflagged) pixels across the entire granule.

    Parameters:
        xds (xr.Dataset): PACE dataset containing Rrs data with dimensions (x, y, wl).
        flags (np.ndarray, optional): Flag array with shape (x, y). Only pixels with
            flags equal to flag_value are used. If None, all non-NaN pixels are used.
        flag_value (int): Flag value indicating valid pixels. Default is 0 (no flags).
        wavelength_subset (np.ndarray, optional): Array of wavelength indices to use.
            If None, all wavelengths are used.
        min_samples (int): Minimum number of valid samples required. Default is 100.

    Returns:
        correlation_matrix (np.ndarray): Correlation matrix with shape (n_wl, n_wl).
        wavelengths (np.ndarray): Array of wavelengths corresponding to matrix dimensions.

    Raises:
        ValueError: If insufficient valid samples are found.
    """
    # Extract Rrs data
    Rrs = xds['Rrs'].values  # Shape: (x, y, wl)
    wavelengths = xds['wavelength'].values

    # Apply wavelength subset if specified
    if wavelength_subset is not None:
        Rrs = Rrs[:, :, wavelength_subset]
        wavelengths = wavelengths[wavelength_subset]

    n_x, n_y, n_wl = Rrs.shape

    # Determine valid pixels
    if flags is not None:
        valid_mask = (flags == flag_value)
    else:
        # Use pixels where at least some wavelengths have valid data
        valid_mask = np.any(np.isfinite(Rrs), axis=2)

    # Extract valid pixels and reshape
    # Shape: (n_valid_pixels, n_wl)
    Rrs_valid = Rrs[valid_mask, :]

    # Additional filtering: remove pixels with too many NaNs
    # Keep pixels with at least 50% valid wavelengths
    valid_fraction = np.sum(np.isfinite(Rrs_valid), axis=1) / n_wl
    good_pixels = valid_fraction > 0.5
    Rrs_valid = Rrs_valid[good_pixels, :]

    n_samples = Rrs_valid.shape[0]

    if n_samples < min_samples:
        raise ValueError(
            f"Insufficient valid samples: {n_samples} < {min_samples}. "
            "Consider relaxing flag criteria or using a different granule."
        )

    # Calculate correlation matrix
    # Use numpy's corrcoef with rowvar=False (each column is a variable/wavelength)
    # This handles NaN values properly
    correlation_matrix = np.corrcoef(Rrs_valid, rowvar=False)

    # Check for NaN in correlation matrix
    nan_fraction = np.sum(np.isnan(correlation_matrix)) / correlation_matrix.size
    if nan_fraction > 0:
        warnings.warn(
            f"{nan_fraction*100:.1f}% of correlation matrix contains NaN values. "
            "This may be due to wavelengths with insufficient valid data."
        )

    return correlation_matrix, wavelengths


def calc_local_correlation(
    xds: xr.Dataset,
    window_size: int = 10,
    flags: Optional[np.ndarray] = None,
    flag_value: int = 0,
    wavelength_subset: Optional[np.ndarray] = None,
    min_samples: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate local correlation matrices for each pixel using spatial neighborhoods.

    This function computes correlation matrices using pixels within a spatial window
    around each pixel. This can capture spatial variability in correlations.

    Parameters:
        xds (xr.Dataset): PACE dataset containing Rrs data with dimensions (x, y, wl).
        window_size (int): Size of spatial window (in pixels) for local correlation.
            Window will be (window_size x window_size) centered on each pixel.
        flags (np.ndarray, optional): Flag array with shape (x, y).
        flag_value (int): Flag value indicating valid pixels. Default is 0.
        wavelength_subset (np.ndarray, optional): Array of wavelength indices to use.
        min_samples (int): Minimum valid samples in window. Default is 20.

    Returns:
        correlation_matrices (np.ndarray): Array of correlation matrices with shape
            (x, y, n_wl, n_wl). Pixels with insufficient samples have NaN matrices.
        wavelengths (np.ndarray): Array of wavelengths.

    Note:
        This function can be memory-intensive for large granules. Consider using
        a subset of the data or processing in chunks.
    """
    # Extract Rrs data
    Rrs = xds['Rrs'].values  # Shape: (x, y, wl)
    wavelengths = xds['wavelength'].values

    # Apply wavelength subset if specified
    if wavelength_subset is not None:
        Rrs = Rrs[:, :, wavelength_subset]
        wavelengths = wavelengths[wavelength_subset]

    n_x, n_y, n_wl = Rrs.shape
    half_window = window_size // 2

    # Initialize output array
    correlation_matrices = np.full((n_x, n_y, n_wl, n_wl), np.nan)

    # Iterate over each pixel
    for i in range(n_x):
        for j in range(n_y):
            # Define window boundaries
            i_min = max(0, i - half_window)
            i_max = min(n_x, i + half_window + 1)
            j_min = max(0, j - half_window)
            j_max = min(n_y, j + half_window + 1)

            # Extract window
            window_Rrs = Rrs[i_min:i_max, j_min:j_max, :]

            # Apply flags if provided
            if flags is not None:
                window_flags = flags[i_min:i_max, j_min:j_max]
                valid_mask = (window_flags == flag_value)
                window_Rrs = window_Rrs[valid_mask, :]
            else:
                # Reshape to (n_pixels, n_wl)
                window_Rrs = window_Rrs.reshape(-1, n_wl)

            # Filter pixels with sufficient valid wavelengths
            valid_fraction = np.sum(np.isfinite(window_Rrs), axis=1) / n_wl
            good_pixels = valid_fraction > 0.5
            window_Rrs = window_Rrs[good_pixels, :]

            # Check if we have enough samples
            if window_Rrs.shape[0] < min_samples:
                continue

            # Calculate correlation matrix for this window
            try:
                corr_matrix = np.corrcoef(window_Rrs, rowvar=False)
                correlation_matrices[i, j, :, :] = corr_matrix
            except:
                # Skip if correlation calculation fails
                continue

    return correlation_matrices, wavelengths


def calc_error_covariance_matrices(
    xds: xr.Dataset,
    correlation_matrix: Optional[np.ndarray] = None,
    correlation_matrices: Optional[np.ndarray] = None,
    wavelength_subset: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate error covariance matrices for each pixel using Rrs uncertainties.

    The error covariance matrix is calculated as:
        Cov[i,j] = Corr[i,j] * sigma[i] * sigma[j]

    where Corr is the correlation matrix, and sigma are the Rrs uncertainties.

    Parameters:
        xds (xr.Dataset): PACE dataset containing 'Rrs_unc' data.
        correlation_matrix (np.ndarray, optional): Global correlation matrix with
            shape (n_wl, n_wl). Used if correlation_matrices is not provided.
        correlation_matrices (np.ndarray, optional): Local correlation matrices with
            shape (x, y, n_wl, n_wl). Takes precedence over correlation_matrix.
        wavelength_subset (np.ndarray, optional): Wavelength indices to use.

    Returns:
        covariance_matrices (np.ndarray): Error covariance matrices with shape
            (x, y, n_wl, n_wl).
        wavelengths (np.ndarray): Array of wavelengths.

    Raises:
        ValueError: If neither correlation_matrix nor correlation_matrices is provided.
    """
    if correlation_matrix is None and correlation_matrices is None:
        raise ValueError(
            "Either correlation_matrix or correlation_matrices must be provided."
        )

    # Extract Rrs uncertainties
    Rrs_unc = xds['Rrs_unc'].values  # Shape: (x, y, wl)
    wavelengths = xds['wavelength'].values

    # Apply wavelength subset if specified
    if wavelength_subset is not None:
        Rrs_unc = Rrs_unc[:, :, wavelength_subset]
        wavelengths = wavelengths[wavelength_subset]

    n_x, n_y, n_wl = Rrs_unc.shape

    # Initialize output array
    covariance_matrices = np.full((n_x, n_y, n_wl, n_wl), np.nan)

    # Use global or local correlations
    use_local = correlation_matrices is not None

    # Iterate over each pixel
    for i in range(n_x):
        for j in range(n_y):
            # Get uncertainties for this pixel
            sigma = Rrs_unc[i, j, :]

            # Skip if uncertainties are invalid
            if not np.all(np.isfinite(sigma)):
                continue

            # Get correlation matrix for this pixel
            if use_local:
                corr = correlation_matrices[i, j, :, :]
                if not np.all(np.isfinite(corr)):
                    continue
            else:
                corr = correlation_matrix

            # Calculate covariance matrix: Cov[i,j] = Corr[i,j] * sigma[i] * sigma[j]
            # Using outer product: sigma.reshape(-1,1) @ sigma.reshape(1,-1)
            sigma_outer = np.outer(sigma, sigma)
            cov = corr * sigma_outer

            covariance_matrices[i, j, :, :] = cov

    return covariance_matrices, wavelengths


def process_granule(
    filename: str,
    method: str = 'global',
    window_size: int = 10,
    full_flag: bool = False,
    wavelength_subset: Optional[np.ndarray] = None,
    save_results: bool = False,
    output_file: Optional[str] = None
) -> dict:
    """
    Complete pipeline for processing PACE granule and generating error covariance matrices.

    This is a convenience function that handles the full workflow:
    1. Load PACE granule
    2. Calculate correlation matrices
    3. Generate error covariance matrices

    Parameters:
        filename (str): Path to PACE L2 netCDF file.
        method (str): Correlation calculation method. Options:
            - 'global': Use global correlation from entire granule (default)
            - 'local': Use local correlations from spatial neighborhoods
        window_size (int): Window size for local correlation method. Default is 10.
        full_flag (bool): If True, only use pixels with no flags. Default is False.
        wavelength_subset (np.ndarray, optional): Indices of wavelengths to use.
            If None, all wavelengths are used.
        save_results (bool): If True, save results to file. Default is False.
        output_file (str, optional): Output file path. If None and save_results=True,
            generates filename based on input.

    Returns:
        results (dict): Dictionary containing:
            - 'xds': xarray Dataset with Rrs data
            - 'flags': Flag array
            - 'wavelengths': Wavelength array
            - 'correlation_matrix': Global correlation matrix (if method='global')
            - 'correlation_matrices': Local correlation matrices (if method='local')
            - 'covariance_matrices': Error covariance matrices for each pixel

    Example:
        >>> results = process_granule('PACE_OCI.20240416T093158.L2.OC_AOP.nc')
        >>> cov = results['covariance_matrices']
        >>> # Access error covariance matrix for pixel (100, 200)
        >>> pixel_cov = cov[100, 200, :, :]
    """
    from ocpy.pace import io as pace_io

    # Load PACE data
    print(f"Loading PACE granule: {filename}")
    xds, flags = pace_io.load_oci_l2(filename, full_flag=full_flag)

    wavelengths = xds['wavelength'].values
    if wavelength_subset is not None:
        wavelengths = wavelengths[wavelength_subset]

    print(f"Granule dimensions: {xds['Rrs'].shape}")
    print(f"Number of wavelengths: {len(wavelengths)}")

    # Calculate correlation matrices
    results = {
        'xds': xds,
        'flags': flags,
        'wavelengths': wavelengths
    }

    if method == 'global':
        print("Calculating global correlation matrix...")
        correlation_matrix, _ = calc_global_correlation(
            xds,
            flags=flags if not full_flag else None,
            flag_value=0,
            wavelength_subset=wavelength_subset
        )
        results['correlation_matrix'] = correlation_matrix

        print(f"Correlation matrix shape: {correlation_matrix.shape}")
        print(f"Correlation range: [{np.nanmin(correlation_matrix):.3f}, "
              f"{np.nanmax(correlation_matrix):.3f}]")

        # Generate error covariance matrices
        print("Generating error covariance matrices...")
        covariance_matrices, _ = calc_error_covariance_matrices(
            xds,
            correlation_matrix=correlation_matrix,
            wavelength_subset=wavelength_subset
        )

    elif method == 'local':
        print(f"Calculating local correlation matrices (window={window_size})...")
        correlation_matrices, _ = calc_local_correlation(
            xds,
            window_size=window_size,
            flags=flags if not full_flag else None,
            flag_value=0,
            wavelength_subset=wavelength_subset
        )
        results['correlation_matrices'] = correlation_matrices

        # Count valid correlation matrices
        n_valid = np.sum(np.all(np.isfinite(correlation_matrices), axis=(2, 3)))
        print(f"Valid correlation matrices: {n_valid} / "
              f"{correlation_matrices.shape[0] * correlation_matrices.shape[1]}")

        # Generate error covariance matrices
        print("Generating error covariance matrices...")
        covariance_matrices, _ = calc_error_covariance_matrices(
            xds,
            correlation_matrices=correlation_matrices,
            wavelength_subset=wavelength_subset
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'global' or 'local'.")

    results['covariance_matrices'] = covariance_matrices

    # Count valid covariance matrices
    n_valid_cov = np.sum(np.all(np.isfinite(covariance_matrices), axis=(2, 3)))
    print(f"Valid covariance matrices: {n_valid_cov} / "
          f"{covariance_matrices.shape[0] * covariance_matrices.shape[1]}")

    # Save results if requested
    if save_results:
        if output_file is None:
            output_file = filename.replace('.nc', '_covariance.npz')

        print(f"Saving results to: {output_file}")
        np.savez_compressed(
            output_file,
            covariance_matrices=covariance_matrices,
            wavelengths=wavelengths,
            correlation_matrix=results.get('correlation_matrix', None),
            method=method
        )

    print("Processing complete!")
    return results


def extract_pixel_covariance(
    covariance_matrices: np.ndarray,
    x: int,
    y: int,
    check_valid: bool = True
) -> np.ndarray:
    """
    Extract error covariance matrix for a specific pixel.

    Parameters:
        covariance_matrices (np.ndarray): Array of covariance matrices with shape
            (n_x, n_y, n_wl, n_wl).
        x (int): X-coordinate of pixel.
        y (int): Y-coordinate of pixel.
        check_valid (bool): If True, raises error if covariance matrix contains NaN.

    Returns:
        cov_matrix (np.ndarray): Covariance matrix for the specified pixel with
            shape (n_wl, n_wl).

    Raises:
        ValueError: If pixel coordinates are out of bounds.
        ValueError: If covariance matrix is invalid (contains NaN) and check_valid=True.
    """
    n_x, n_y = covariance_matrices.shape[:2]

    if x < 0 or x >= n_x or y < 0 or y >= n_y:
        raise ValueError(
            f"Pixel coordinates ({x}, {y}) out of bounds. "
            f"Valid range: x=[0, {n_x}), y=[0, {n_y})"
        )

    cov_matrix = covariance_matrices[x, y, :, :]

    if check_valid and not np.all(np.isfinite(cov_matrix)):
        raise ValueError(
            f"Invalid covariance matrix for pixel ({x}, {y}). "
            "Contains NaN or Inf values."
        )

    return cov_matrix


def get_correlation_summary(correlation_matrix: np.ndarray) -> dict:
    """
    Generate summary statistics for a correlation matrix.

    Parameters:
        correlation_matrix (np.ndarray): Correlation matrix with shape (n_wl, n_wl).

    Returns:
        summary (dict): Dictionary containing:
            - 'mean': Mean correlation (excluding diagonal)
            - 'std': Standard deviation of correlations
            - 'min': Minimum correlation
            - 'max': Maximum correlation (excluding diagonal)
            - 'n_negative': Number of negative correlations
    """
    # Extract off-diagonal elements
    n = correlation_matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    off_diag = correlation_matrix[mask]

    # Remove NaN values
    off_diag = off_diag[np.isfinite(off_diag)]

    summary = {
        'mean': np.mean(off_diag),
        'std': np.std(off_diag),
        'min': np.min(off_diag),
        'max': np.max(off_diag),
        'n_negative': np.sum(off_diag < 0)
    }

    return summary
