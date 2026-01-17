#!/usr/bin/env python
"""
Command-line script to plot PACE OCI Rrs spectrum with uncertainty.

Extracts Rrs and Rrs_unc at a specified pixel location (x,y or lat/lon)
and displays an interactive plot in the browser using Bokeh.

Usage:
    python plot_pace_rrs.py <granule_file> --lat <lat> --lon <lon>
    python plot_pace_rrs.py <granule_file> --x <x> --y <y>

Examples:
    python plot_pace_rrs.py PACE_OCI.20240401.L2.OC.nc --lat 34.5 --lon -120.3
    python plot_pace_rrs.py PACE_OCI.20240401.L2.OC.nc --x 500 --y 800
"""

import argparse
import sys
import numpy as np

from bokeh.plotting import figure, show
from bokeh.models import Whisker, ColumnDataSource, HoverTool
from bokeh.io import output_file

from ocpy.pace import io as pace_io


def find_nearest_pixel(xds, lat: float, lon: float) -> tuple[int, int]:
    """
    Find the nearest pixel indices (x, y) to the given lat/lon.

    Parameters
    ----------
    xds : xarray.Dataset
        PACE dataset with latitude and longitude coordinates
    lat : float
        Target latitude
    lon : float
        Target longitude

    Returns
    -------
    tuple[int, int]
        (x, y) pixel indices
    """
    lat_arr = xds.latitude.values
    lon_arr = xds.longitude.values

    # Calculate distance to target location
    dist = np.sqrt((lat_arr - lat)**2 + (lon_arr - lon)**2)

    # Find minimum distance index
    min_idx = np.unravel_index(np.nanargmin(dist), dist.shape)
    x_idx, y_idx = int(min_idx[0]), int(min_idx[1])

    # Report actual lat/lon at found pixel
    actual_lat = lat_arr[x_idx, y_idx]
    actual_lon = lon_arr[x_idx, y_idx]
    distance_deg = dist[x_idx, y_idx]

    print(f"Requested: lat={lat:.4f}, lon={lon:.4f}")
    print(f"Found pixel ({x_idx}, {y_idx}): lat={actual_lat:.4f}, lon={actual_lon:.4f}")
    print(f"Distance: {distance_deg:.4f} degrees")

    return x_idx, y_idx


def extract_spectrum(xds, x: int, y: int, flags: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract Rrs spectrum and uncertainty at a pixel location.

    Parameters
    ----------
    xds : xarray.Dataset
        PACE dataset
    x : int
        x pixel index
    y : int
        y pixel index
    flags : np.ndarray
        L2 quality flags

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (wavelength, Rrs, Rrs_unc) arrays
    """
    # Check flag at this location
    flag_val = flags[x, y]
    if flag_val != 0:
        print(f"Warning: Pixel has non-zero quality flag: {flag_val}")

    wavelength = xds.wavelength.values
    rrs = xds.Rrs.values[x, y, :]
    rrs_unc = xds.Rrs_unc.values[x, y, :]

    # Check for fill values / NaNs
    valid = np.isfinite(rrs) & np.isfinite(rrs_unc)
    n_valid = np.sum(valid)

    if n_valid == 0:
        print("Error: No valid Rrs data at this location")
        sys.exit(1)
    elif n_valid < len(wavelength):
        print(f"Note: {len(wavelength) - n_valid} wavelengths have invalid data")

    return wavelength, rrs, rrs_unc


def plot_rrs_spectrum(wavelength: np.ndarray, rrs: np.ndarray, rrs_unc: np.ndarray,
                      x: int, y: int, lat: float = None, lon: float = None,
                      output_html: str = None):
    """
    Create an interactive Bokeh plot of Rrs spectrum with error bars.

    Parameters
    ----------
    wavelength : np.ndarray
        Wavelength values in nm
    rrs : np.ndarray
        Rrs values in sr^-1
    rrs_unc : np.ndarray
        Rrs uncertainty values in sr^-1
    x : int
        x pixel index
    y : int
        y pixel index
    lat : float, optional
        Latitude of pixel
    lon : float, optional
        Longitude of pixel
    output_html : str, optional
        Path to save HTML file (if None, uses temp file)
    """
    # Filter to valid data
    valid = np.isfinite(rrs) & np.isfinite(rrs_unc)
    wl = wavelength[valid]
    r = rrs[valid]
    r_unc = rrs_unc[valid]

    # Create data source for Bokeh
    source = ColumnDataSource(data=dict(
        wavelength=wl,
        rrs=r,
        rrs_unc=r_unc,
        upper=r + r_unc,
        lower=r - r_unc,
    ))

    # Create title
    if lat is not None and lon is not None:
        title = f"PACE OCI Rrs Spectrum - Pixel ({x}, {y}) at ({lat:.3f}°, {lon:.3f}°)"
    else:
        title = f"PACE OCI Rrs Spectrum - Pixel ({x}, {y})"

    # Set output file
    if output_html:
        output_file(output_html, title="PACE Rrs Spectrum")

    # Create figure
    p = figure(
        title=title,
        x_axis_label="Wavelength (nm)",
        y_axis_label="Rrs (sr⁻¹)",
        width=900,
        height=500,
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )

    # Add error bars using Whisker
    whisker = Whisker(
        source=source,
        base="wavelength",
        upper="upper",
        lower="lower",
        level="underlay",
        line_color="gray",
        line_alpha=0.6,
        line_width=1,
    )
    whisker.upper_head.size = 0  # No caps on whiskers for cleaner look
    whisker.lower_head.size = 0
    p.add_layout(whisker)

    # Add line and points
    p.line("wavelength", "rrs", source=source, line_width=1.5, color="navy", alpha=0.8)
    p.scatter("wavelength", "rrs", source=source, size=5, color="navy", alpha=0.8)

    # Add hover tool
    hover = HoverTool(
        tooltips=[
            ("Wavelength", "@wavelength{0.0} nm"),
            ("Rrs", "@rrs{0.0000} sr⁻¹"),
            ("Uncertainty", "±@rrs_unc{0.0000} sr⁻¹"),
        ],
        mode="vline",
    )
    p.add_tools(hover)

    # Style adjustments
    p.title.text_font_size = "14pt"
    p.xaxis.axis_label_text_font_size = "12pt"
    p.yaxis.axis_label_text_font_size = "12pt"
    p.xaxis.major_label_text_font_size = "10pt"
    p.yaxis.major_label_text_font_size = "10pt"

    # Show in browser
    show(p)

    print(f"\nPlot displayed in browser")
    if output_html:
        print(f"HTML saved to: {output_html}")


def parser() -> argparse.ArgumentParser:
    """
    Build and return the argument parser for plot_rrs.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser
    """
    p = argparse.ArgumentParser(
        description="Plot PACE OCI Rrs spectrum with uncertainty at a pixel location",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s granule.nc --lat 34.5 --lon -120.3
  %(prog)s granule.nc --x 500 --y 800
  %(prog)s granule.nc --lat 34.5 --lon -120.3 -o spectrum.html
        """
    )

    p.add_argument("granule", help="Path to PACE OCI L2 netCDF file")

    # Location specification (mutually exclusive groups)
    loc_group = p.add_argument_group("Location (specify either lat/lon OR x/y)")
    loc_group.add_argument("--lat", type=float, help="Latitude (decimal degrees)")
    loc_group.add_argument("--lon", type=float, help="Longitude (decimal degrees)")
    loc_group.add_argument("--x", type=int, help="X pixel index (along-track)")
    loc_group.add_argument("--y", type=int, help="Y pixel index (cross-track)")

    p.add_argument("-o", "--output", help="Output HTML file path (optional)")

    return p


def main(args: argparse.Namespace = None):
    """
    Main entry point for plotting PACE Rrs spectrum.

    Parameters
    ----------
    args : argparse.Namespace, optional
        Parsed arguments. If None, parses from command line.
    """
    if args is None:
        p = parser()
        args = p.parse_args()
    else:
        p = parser()

    # Validate location arguments
    have_latlon = args.lat is not None and args.lon is not None
    have_xy = args.x is not None and args.y is not None

    if not have_latlon and not have_xy:
        p.error("Must specify either --lat and --lon, or --x and --y")

    if have_latlon and have_xy:
        p.error("Specify either lat/lon OR x/y, not both")

    if (args.lat is None) != (args.lon is None):
        p.error("Both --lat and --lon must be specified together")

    if (args.x is None) != (args.y is None):
        p.error("Both --x and --y must be specified together")

    # Load the granule
    print(f"Loading: {args.granule}")
    try:
        xds, flags = pace_io.load_oci_l2(args.granule)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    print(f"Data dimensions: {dict(xds.dims)}")
    print(f"Wavelength range: {xds.wavelength.values[0]}-{xds.wavelength.values[-1]} nm")

    # Determine pixel location
    if have_latlon:
        x_idx, y_idx = find_nearest_pixel(xds, args.lat, args.lon)
        lat_val = float(xds.latitude.values[x_idx, y_idx])
        lon_val = float(xds.longitude.values[x_idx, y_idx])
    else:
        x_idx, y_idx = args.x, args.y
        # Validate indices
        if x_idx < 0 or x_idx >= xds.dims['x']:
            print(f"Error: x index {x_idx} out of range [0, {xds.dims['x']-1}]")
            sys.exit(1)
        if y_idx < 0 or y_idx >= xds.dims['y']:
            print(f"Error: y index {y_idx} out of range [0, {xds.dims['y']-1}]")
            sys.exit(1)
        lat_val = float(xds.latitude.values[x_idx, y_idx])
        lon_val = float(xds.longitude.values[x_idx, y_idx])
        print(f"Pixel ({x_idx}, {y_idx}): lat={lat_val:.4f}, lon={lon_val:.4f}")

    # Extract spectrum
    wavelength, rrs, rrs_unc = extract_spectrum(xds, x_idx, y_idx, flags)

    # Print summary statistics
    valid = np.isfinite(rrs)
    print(f"\nRrs statistics (valid wavelengths: {np.sum(valid)}):")
    print(f"  Min: {np.nanmin(rrs):.6f} sr⁻¹")
    print(f"  Max: {np.nanmax(rrs):.6f} sr⁻¹")
    print(f"  Mean uncertainty: {np.nanmean(rrs_unc):.6f} sr⁻¹")

    # Create plot
    plot_rrs_spectrum(
        wavelength, rrs, rrs_unc,
        x_idx, y_idx,
        lat=lat_val, lon=lon_val,
        output_html=args.output
    )


if __name__ == "__main__":
    main()
