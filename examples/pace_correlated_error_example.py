"""
Example script for processing PACE granules and calculating correlated errors.

This script demonstrates how to use the correlated_error module to:
1. Load PACE OCI L2 data
2. Calculate correlation matrices between Rrs wavelengths
3. Generate pixel-by-pixel error covariance matrices
4. Visualize and analyze the results

Requirements:
    - PACE OCI L2 netCDF file (download from https://oceandata.sci.gsfc.nasa.gov/)
    - Set OS_COLOR environment variable or update file path below

Author: ocpy team
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import colors

from ocpy.pace import correlated_error as ce


def example_global_correlation(pace_file):
    """
    Example 1: Calculate global correlation matrix and generate covariance matrices.

    This is the recommended approach for most applications. It's fast and provides
    consistent error estimates across the granule.
    """
    print("=" * 70)
    print("Example 1: Global Correlation Method")
    print("=" * 70)

    # Use the convenience function to process the entire granule
    results = ce.process_granule(
        pace_file,
        method='global',
        full_flag=False,  # Use all pixels (not just fully valid ones)
        save_results=True  # Save results to npz file
    )

    # Extract results
    correlation_matrix = results['correlation_matrix']
    covariance_matrices = results['covariance_matrices']
    wavelengths = results['wavelengths']

    # Print correlation summary
    summary = ce.get_correlation_summary(correlation_matrix)
    print("\nCorrelation Matrix Summary:")
    print(f"  Mean correlation: {summary['mean']:.3f}")
    print(f"  Std deviation:    {summary['std']:.3f}")
    print(f"  Range:            [{summary['min']:.3f}, {summary['max']:.3f}]")
    print(f"  Negative corr:    {summary['n_negative']}")

    # Visualize correlation matrix
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot correlation matrix
    im1 = axes[0].imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0].set_title('Wavelength Correlation Matrix')
    axes[0].set_xlabel('Wavelength Index')
    axes[0].set_ylabel('Wavelength Index')
    plt.colorbar(im1, ax=axes[0], label='Correlation')

    # Plot correlation vs wavelength separation
    n_wl = len(wavelengths)
    separations = []
    correlations = []
    for i in range(n_wl):
        for j in range(i+1, n_wl):
            sep = wavelengths[j] - wavelengths[i]
            corr = correlation_matrix[i, j]
            if np.isfinite(corr):
                separations.append(sep)
                correlations.append(corr)

    axes[1].scatter(separations, correlations, alpha=0.3, s=5)
    axes[1].set_xlabel('Wavelength Separation (nm)')
    axes[1].set_ylabel('Correlation')
    axes[1].set_title('Correlation vs Wavelength Separation')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(0, color='k', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('pace_correlation_matrix.png', dpi=150)
    print("\nSaved correlation visualization to: pace_correlation_matrix.png")

    return results


def example_pixel_analysis(results, x=500, y=600):
    """
    Example 2: Analyze error covariance for a specific pixel.

    This shows how to extract and use the error covariance matrix for
    uncertainty propagation in downstream analyses.
    """
    print("\n" + "=" * 70)
    print(f"Example 2: Single Pixel Analysis (x={x}, y={y})")
    print("=" * 70)

    covariance_matrices = results['covariance_matrices']
    wavelengths = results['wavelengths']
    xds = results['xds']

    # Extract data for this pixel
    try:
        pixel_cov = ce.extract_pixel_covariance(covariance_matrices, x, y)
        pixel_rrs = xds['Rrs'].values[x, y, :]
        pixel_unc = xds['Rrs_unc'].values[x, y, :]
    except (ValueError, IndexError) as e:
        print(f"Error extracting pixel data: {e}")
        print("Trying a different pixel location...")
        # Find a valid pixel
        valid_pixels = np.where(np.all(np.isfinite(covariance_matrices), axis=(2, 3)))
        if len(valid_pixels[0]) > 0:
            x, y = valid_pixels[0][0], valid_pixels[1][0]
            print(f"Using pixel ({x}, {y}) instead")
            pixel_cov = ce.extract_pixel_covariance(covariance_matrices, x, y)
            pixel_rrs = xds['Rrs'].values[x, y, :]
            pixel_unc = xds['Rrs_unc'].values[x, y, :]
        else:
            print("No valid pixels found!")
            return

    # Calculate correlation from covariance
    # Corr[i,j] = Cov[i,j] / (sigma[i] * sigma[j])
    sigma_outer = np.outer(pixel_unc, pixel_unc)
    pixel_corr = pixel_cov / (sigma_outer + 1e-20)  # Add small epsilon to avoid division by zero

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot Rrs spectrum with uncertainties
    axes[0, 0].errorbar(wavelengths, pixel_rrs, yerr=pixel_unc,
                        fmt='o-', capsize=3, markersize=4)
    axes[0, 0].set_xlabel('Wavelength (nm)')
    axes[0, 0].set_ylabel('Rrs (sr$^{-1}$)')
    axes[0, 0].set_title(f'Rrs Spectrum (Pixel {x}, {y})')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot uncertainties
    axes[0, 1].plot(wavelengths, pixel_unc, 'o-')
    axes[0, 1].set_xlabel('Wavelength (nm)')
    axes[0, 1].set_ylabel('Rrs Uncertainty (sr$^{-1}$)')
    axes[0, 1].set_title('Rrs Uncertainties')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot correlation matrix for this pixel
    im = axes[1, 0].imshow(pixel_corr, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1, 0].set_title('Correlation Matrix')
    axes[1, 0].set_xlabel('Wavelength Index')
    axes[1, 0].set_ylabel('Wavelength Index')
    plt.colorbar(im, ax=axes[1, 0], label='Correlation')

    # Plot covariance matrix
    im2 = axes[1, 1].imshow(pixel_cov, cmap='viridis',
                            norm=colors.SymLogNorm(linthresh=1e-8, vmin=pixel_cov.min(), vmax=pixel_cov.max()))
    axes[1, 1].set_title('Covariance Matrix')
    axes[1, 1].set_xlabel('Wavelength Index')
    axes[1, 1].set_ylabel('Wavelength Index')
    plt.colorbar(im2, ax=axes[1, 1], label='Covariance')

    plt.tight_layout()
    plt.savefig('pace_pixel_error_analysis.png', dpi=150)
    print(f"\nSaved pixel analysis to: pace_pixel_error_analysis.png")

    # Print some statistics
    print(f"\nPixel Statistics:")
    print(f"  Mean Rrs: {np.nanmean(pixel_rrs):.6f} sr^-1")
    print(f"  Mean uncertainty: {np.nanmean(pixel_unc):.6f} sr^-1")
    print(f"  Mean correlation: {np.nanmean(pixel_corr[~np.eye(len(pixel_corr), dtype=bool)]):.3f}")

    # Calculate total variance for a simple sum (useful for band ratios, etc.)
    # Var(X + Y) = Var(X) + Var(Y) + 2*Cov(X,Y)
    total_variance = np.sum(pixel_cov)
    total_std = np.sqrt(total_variance)
    print(f"  Total std (sum of all wavelengths): {total_std:.6f} sr^-1")


def example_spatial_variability(results):
    """
    Example 3: Visualize spatial variability in uncertainties.

    This shows how uncertainties vary across the granule.
    """
    print("\n" + "=" * 70)
    print("Example 3: Spatial Variability of Uncertainties")
    print("=" * 70)

    xds = results['xds']
    covariance_matrices = results['covariance_matrices']
    wavelengths = results['wavelengths']

    # Calculate mean uncertainty at each pixel (averaged over wavelengths)
    mean_unc = np.nanmean(xds['Rrs_unc'].values, axis=2)

    # Calculate determinant of covariance matrices (measure of total uncertainty)
    n_x, n_y = covariance_matrices.shape[:2]
    cov_det = np.full((n_x, n_y), np.nan)

    print("Calculating covariance determinants...")
    for i in range(n_x):
        if i % 100 == 0:
            print(f"  Processing row {i}/{n_x}...")
        for j in range(n_y):
            cov = covariance_matrices[i, j, :, :]
            if np.all(np.isfinite(cov)):
                try:
                    cov_det[i, j] = np.linalg.det(cov)
                except:
                    pass

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot mean uncertainty
    im1 = axes[0].imshow(mean_unc, cmap='viridis', aspect='auto')
    axes[0].set_title('Mean Rrs Uncertainty')
    axes[0].set_xlabel('Y Pixel')
    axes[0].set_ylabel('X Pixel')
    plt.colorbar(im1, ax=axes[0], label='Mean Rrs Unc (sr$^{-1}$)')

    # Plot covariance determinant (log scale)
    cov_det_pos = np.abs(cov_det)  # Use absolute value for log scale
    im2 = axes[1].imshow(cov_det_pos, cmap='plasma', aspect='auto',
                        norm=colors.LogNorm(vmin=np.nanpercentile(cov_det_pos, 1),
                                           vmax=np.nanpercentile(cov_det_pos, 99)))
    axes[1].set_title('Covariance Matrix Determinant')
    axes[1].set_xlabel('Y Pixel')
    axes[1].set_ylabel('X Pixel')
    plt.colorbar(im2, ax=axes[1], label='|Det(Cov)|')

    plt.tight_layout()
    plt.savefig('pace_spatial_uncertainty.png', dpi=150)
    print("\nSaved spatial visualization to: pace_spatial_uncertainty.png")


def example_wavelength_subset():
    """
    Example 4: Process only a subset of wavelengths.

    This is useful for reducing computation time when you only need
    specific wavelengths.
    """
    print("\n" + "=" * 70)
    print("Example 4: Wavelength Subset Processing")
    print("=" * 70)

    # Define subset: select every 10th wavelength
    subset_indices = np.arange(0, 184, 10)

    print(f"Processing {len(subset_indices)} wavelengths (every 10th)")

    # Process with subset
    results = ce.process_granule(
        pace_file,
        method='global',
        wavelength_subset=subset_indices,
        save_results=False
    )

    print(f"Processed wavelengths: {results['wavelengths']}")

    return results


if __name__ == "__main__":
    # Set up file path
    # Option 1: Use environment variable
    if os.getenv('OS_COLOR') is not None:
        pace_file = os.path.join(
            os.getenv('OS_COLOR'),
            'data', 'PACE', 'early',
            'PACE_OCI.20240416T093158.L2.OC_AOP.V1_0_0.NRT.nc'
        )
    else:
        # Option 2: Specify path directly
        pace_file = 'PACE_OCI.20240416T093158.L2.OC_AOP.V1_0_0.NRT.nc'

    # Check if file exists
    if not os.path.exists(pace_file):
        print(f"ERROR: PACE file not found: {pace_file}")
        print("\nPlease either:")
        print("  1. Set OS_COLOR environment variable to point to your data directory")
        print("  2. Update the pace_file variable in this script")
        print("  3. Download PACE data from: https://oceandata.sci.gsfc.nasa.gov/")
        exit(1)

    print(f"Processing PACE file: {pace_file}\n")

    # Run examples
    try:
        # Example 1: Global correlation (recommended)
        results = example_global_correlation(pace_file)

        # Example 2: Analyze specific pixel
        example_pixel_analysis(results, x=500, y=600)

        # Example 3: Spatial variability
        example_spatial_variability(results)

        # Example 4: Wavelength subset (commented out to save time)
        # example_wavelength_subset()

        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
