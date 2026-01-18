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
import os
import sys
import numpy as np

from bokeh.plotting import figure, show
from bokeh.models import Whisker, ColumnDataSource, HoverTool, Span, Button, CustomJS
from bokeh.layouts import column
from bokeh.io import output_file

from ocpy.pace import io as pace_io


def plot_rrs_spectrum(wavelength: np.ndarray, rrs: np.ndarray, rrs_unc: np.ndarray,
                      x: int, y: int, lat: float = None, lon: float = None,
                      output_html: str = None, filename: str = None):
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
    filename : str, optional
        Name of the source data file to display in title
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
    if filename:
        title = f"{filename}\n{title}"

    # Set output file
    if output_html:
        output_file(output_html, title="PACE Rrs Spectrum")

    # Compute y-axis ranges for linear and log modes
    y_min_linear = float(np.min(r - r_unc))
    y_max_linear = float(np.max(r + r_unc))
    y_padding = (y_max_linear - y_min_linear) * 0.05

    # For log scale, find min/max of positive values only
    positive_mask = r > 0
    if np.any(positive_mask):
        y_min_log = float(np.min(r[positive_mask]))
        y_max_log = float(np.max(r[positive_mask]))
    else:
        y_min_log = 1e-6
        y_max_log = 1e-2

    # Common figure properties
    fig_kwargs = dict(
        title=title,
        x_axis_label="Wavelength (nm)",
        y_axis_label="Rrs (sr⁻¹)",
        width=900,
        height=500,
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )

    # Create LINEAR figure
    p_linear = figure(**fig_kwargs)

    # Add horizontal line at Rrs=0 (linear only)
    zero_line = Span(location=0, dimension='width', line_color='black',
                     line_dash='dashed', line_width=1)
    p_linear.add_layout(zero_line)

    # Add error bars (linear only)
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
    whisker.upper_head.size = 0
    whisker.lower_head.size = 0
    p_linear.add_layout(whisker)

    # Add points
    p_linear.scatter("wavelength", "rrs", source=source, size=5, color="navy", alpha=0.8)

    # Add hover tool
    hover_linear = HoverTool(
        tooltips=[
            ("Wavelength", "@wavelength{0.0} nm"),
            ("Rrs", "@rrs{0.0000} sr⁻¹"),
            ("Uncertainty", "±@rrs_unc{0.0000} sr⁻¹"),
        ],
        mode="vline",
    )
    p_linear.add_tools(hover_linear)

    # Create LOG figure
    p_log = figure(y_axis_type="log", **fig_kwargs)
    p_log.y_range.start = y_min_log * 0.5
    p_log.y_range.end = y_max_log * 2.0
    p_log.visible = False  # Start hidden

    # Create separate data source for log plot error bars (only where lower bound > 0)
    log_err_mask = (r - r_unc) > 0
    source_log_err = ColumnDataSource(data=dict(
        wavelength=wl[log_err_mask],
        rrs=r[log_err_mask],
        rrs_unc=r_unc[log_err_mask],
        upper=(r + r_unc)[log_err_mask],
        lower=(r - r_unc)[log_err_mask],
    ))

    # Add error bars to log plot (only for positive lower bounds)
    whisker_log = Whisker(
        source=source_log_err,
        base="wavelength",
        upper="upper",
        lower="lower",
        level="underlay",
        line_color="gray",
        line_alpha=0.6,
        line_width=1,
    )
    whisker_log.upper_head.size = 0
    whisker_log.lower_head.size = 0
    p_log.add_layout(whisker_log)

    # Add points to log plot (use full source for all positive Rrs)
    p_log.scatter("wavelength", "rrs", source=source, size=5, color="navy", alpha=0.8)

    # Add hover tool to log plot
    hover_log = HoverTool(
        tooltips=[
            ("Wavelength", "@wavelength{0.0} nm"),
            ("Rrs", "@rrs{0.0000} sr⁻¹"),
            ("Uncertainty", "±@rrs_unc{0.0000} sr⁻¹"),
        ],
        mode="vline",
    )
    p_log.add_tools(hover_log)

    # Style adjustments for both figures
    for p in [p_linear, p_log]:
        p.title.text_font_size = "14pt"
        p.xaxis.axis_label_text_font_size = "17pt"
        p.yaxis.axis_label_text_font_size = "17pt"
        p.xaxis.major_label_text_font_size = "14pt"
        p.yaxis.major_label_text_font_size = "14pt"

    # Create toggle button with JavaScript callback
    toggle_button = Button(label="Toggle Linear/Log Y-axis", button_type="default")
    toggle_callback = CustomJS(args=dict(
        p_linear=p_linear,
        p_log=p_log,
    ), code="""
        if (p_linear.visible) {
            p_linear.visible = false;
            p_log.visible = true;
        } else {
            p_linear.visible = true;
            p_log.visible = false;
        }
    """)
    toggle_button.js_on_click(toggle_callback)

    # Create layout with button and both plots
    layout = column(toggle_button, p_linear, p_log)

    # Show in browser
    show(layout)

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

    # Load only the single spectrum (much faster than loading full granule)
    print(f"Loading spectrum from: {args.granule}")
    try:
        if have_latlon:
            wavelength, rrs, rrs_unc, flag, pixel_coords = pace_io.load_oci_l2_spectrum(
                args.granule, args.lat, args.lon
            )
            x_idx, y_idx, lat_val, lon_val = pixel_coords
            print(f"Requested: lat={args.lat:.4f}, lon={args.lon:.4f}")
            print(f"Found pixel ({x_idx}, {y_idx}): lat={lat_val:.4f}, lon={lon_val:.4f}")
        else:
            wavelength, rrs, rrs_unc, flag, pixel_coords = pace_io.load_oci_l2_spectrum_pixel(
                args.granule, args.x, args.y
            )
            x_idx, y_idx, lat_val, lon_val = pixel_coords
            print(f"Pixel ({x_idx}, {y_idx}): lat={lat_val:.4f}, lon={lon_val:.4f}")
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    print(f"Wavelength range: {wavelength[0]:.1f}-{wavelength[-1]:.1f} nm")

    # Check flag at this location
    if flag != 0:
        print(f"Warning: Pixel has non-zero quality flag: {flag}")

    # Print summary statistics
    valid = np.isfinite(rrs)
    n_valid = np.sum(valid)
    if n_valid == 0:
        print("Error: No valid Rrs data at this location")
        sys.exit(1)
    elif n_valid < len(wavelength):
        print(f"Note: {len(wavelength) - n_valid} wavelengths have invalid data")

    print(f"\nRrs statistics (valid wavelengths: {n_valid}):")
    print(f"  Min: {np.nanmin(rrs):.6f} sr⁻¹")
    print(f"  Max: {np.nanmax(rrs):.6f} sr⁻¹")
    print(f"  Mean uncertainty: {np.nanmean(rrs_unc):.6f} sr⁻¹")

    # Create plot
    plot_rrs_spectrum(
        wavelength, rrs, rrs_unc,
        x_idx, y_idx,
        lat=lat_val, lon=lon_val,
        output_html=args.output,
        filename=os.path.basename(args.granule)
    )


if __name__ == "__main__":
    main()
