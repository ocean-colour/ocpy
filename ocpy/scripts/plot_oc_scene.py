#!/usr/bin/env python
"""
Command-line script to plot ocean color granule scenes in a multi-panel display.

Displays wavelength channels from PACE (L1C or L2) or MODIS granules as a grid
of sub-panels in the browser using Bokeh. Each panel shows a different wavelength
channel with lat/lon coordinates. Masked pixels are displayed in gray.

Usage:
    plot_oc_scene.py <granule_file> [--channels WAVELENGTHS] [--output FILE]

Examples:
    # PACE L2 with default channels
    plot_oc_scene.py PACE_OCI.20240401.L2.OC.nc

    # PACE L2 with custom channels
    plot_oc_scene.py PACE_OCI.20240401.L2.OC.nc --channels 400,450,550,650

    # PACE L1C granule
    plot_oc_scene.py PACE_OCI.20240401.L1C.nc --channels 400,500,600,700

    # Save to HTML file
    plot_oc_scene.py granule.nc --channels 400,500,600 -o scene_plot.html
"""

import argparse
import os
import sys
import numpy as np
from netCDF4 import Dataset
from typing import Tuple, List, Optional

from bokeh.plotting import figure, show
from bokeh.models import HoverTool, LinearColorMapper, ColorBar, BasicTicker
from bokeh.layouts import gridplot
from bokeh.io import output_file
from bokeh.palettes import Viridis256


def detect_file_type(filename: str) -> str:
    """
    Detect the type of ocean color granule file.

    Parameters
    ----------
    filename : str
        Path to the netCDF file

    Returns
    -------
    str
        File type: 'PACE_L2', 'PACE_L1C', 'MODIS_L2', or 'UNKNOWN'
    """
    try:
        with Dataset(filename, 'r') as ds:
            # Check for PACE by looking at groups and global attributes
            if 'title' in ds.ncattrs():
                title = ds.getncattr('title')
                if 'PACE' in title or 'OCI' in title:
                    if 'L2' in title or 'geophysical_data' in ds.groups:
                        return 'PACE_L2'
                    elif 'L1C' in title or 'observation_data' in ds.groups:
                        return 'PACE_L1C'

            # Check for MODIS
            if 'instrument' in ds.ncattrs():
                instrument = ds.getncattr('instrument')
                if 'MODIS' in instrument:
                    return 'MODIS_L2'

            # Alternative detection methods
            if 'geophysical_data' in ds.groups:
                groups = list(ds.groups.keys())
                if 'sensor_band_parameters' in groups and 'navigation_data' in groups:
                    return 'PACE_L2'

            if 'observation_data' in ds.groups:
                return 'PACE_L1C'

    except Exception as e:
        print(f"Error detecting file type: {e}")

    return 'UNKNOWN'


def load_pace_l2_scene(filename: str, wavelengths: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load PACE L2 scene data for specified wavelengths.

    Parameters
    ----------
    filename : str
        Path to PACE L2 netCDF file
    wavelengths : list of float
        Wavelengths to extract (nm)

    Returns
    -------
    data : np.ndarray
        Data array of shape (n_wavelengths, nx, ny)
    lats : np.ndarray
        Latitude array of shape (nx, ny)
    lons : np.ndarray
        Longitude array of shape (nx, ny)
    flags : np.ndarray
        Flag array of shape (nx, ny)
    actual_wavelengths : np.ndarray
        Actual wavelengths found in file (closest matches)
    """
    with Dataset(filename, 'r') as ds:
        # Access groups
        gd = ds.groups['geophysical_data']
        nav = ds.groups['navigation_data']
        sbp = ds.groups['sensor_band_parameters']

        # Load coordinates
        lats = nav.variables['latitude'][:]
        lons = nav.variables['longitude'][:]
        available_wls = sbp['wavelength_3d'][:]
        flags = gd.variables['l2_flags'][:]

        # Load Rrs data
        rrs = gd.variables['Rrs'][:]  # Shape: (nx, ny, n_wavelengths)

        # Find closest wavelengths
        data_list = []
        actual_wls = []

        for target_wl in wavelengths:
            idx = np.argmin(np.abs(available_wls - target_wl))
            actual_wls.append(available_wls[idx])
            data_list.append(rrs[:, :, idx])

        data = np.array(data_list)  # Shape: (n_wavelengths, nx, ny)
        actual_wavelengths = np.array(actual_wls)

    return data, lats, lons, flags, actual_wavelengths


def load_pace_l1c_scene(filename: str, wavelengths: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load PACE L1C scene data for specified wavelengths.

    Parameters
    ----------
    filename : str
        Path to PACE L1C netCDF file
    wavelengths : list of float
        Wavelengths to extract (nm)

    Returns
    -------
    data : np.ndarray
        Data array of shape (n_wavelengths, nx, ny)
    lats : np.ndarray
        Latitude array of shape (nx, ny)
    lons : np.ndarray
        Longitude array of shape (nx, ny)
    flags : np.ndarray
        Flag array of shape (nx, ny)
    actual_wavelengths : np.ndarray
        Actual wavelengths found in file (closest matches)
    """
    with Dataset(filename, 'r') as ds:
        # Access groups
        obs = ds.groups['observation_data']
        nav = ds.groups['navigation_data']
        sbp = ds.groups['sensor_band_parameters']

        # Load coordinates
        lats = nav.variables['latitude'][:]
        lons = nav.variables['longitude'][:]
        available_wls = sbp['wavelength'][:]

        # Load Lt (top of atmosphere radiance) or rhot (TOA reflectance)
        # Try rhot first (more commonly used), fall back to Lt
        if 'rhot' in obs.variables:
            data_var = obs.variables['rhot'][:]
            data_name = 'rhot'
        elif 'Lt' in obs.variables:
            data_var = obs.variables['Lt'][:]
            data_name = 'Lt'
        else:
            raise ValueError("Could not find 'rhot' or 'Lt' in L1C file")

        # Create a simple flag array (mark invalid/masked as non-zero)
        flags = np.zeros(lats.shape, dtype=np.int32)

        # Find closest wavelengths
        data_list = []
        actual_wls = []

        for target_wl in wavelengths:
            idx = np.argmin(np.abs(available_wls - target_wl))
            actual_wls.append(available_wls[idx])
            channel_data = data_var[:, :, idx]

            # Mark masked/invalid pixels in flags
            flags |= np.where(np.isfinite(channel_data), 0, 1)

            data_list.append(channel_data)

        data = np.array(data_list)  # Shape: (n_wavelengths, nx, ny)
        actual_wavelengths = np.array(actual_wls)

    print(f"Loaded L1C variable: {data_name}")
    return data, lats, lons, flags, actual_wavelengths


def load_modis_l2_scene(filename: str, wavelengths: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load MODIS L2 scene data for specified wavelengths.

    Parameters
    ----------
    filename : str
        Path to MODIS L2 netCDF file
    wavelengths : list of float
        Wavelengths to extract (nm)

    Returns
    -------
    data : np.ndarray
        Data array of shape (n_wavelengths, nx, ny)
    lats : np.ndarray
        Latitude array of shape (nx, ny)
    lons : np.ndarray
        Longitude array of shape (nx, ny)
    flags : np.ndarray
        Flag array of shape (nx, ny)
    actual_wavelengths : np.ndarray
        Actual wavelengths found in file
    """
    # MODIS has fixed bands, not hyperspectral
    modis_bands = {
        412: 'Rrs_412',
        443: 'Rrs_443',
        469: 'Rrs_469',
        488: 'Rrs_488',
        531: 'Rrs_531',
        547: 'Rrs_547',
        555: 'Rrs_555',
        645: 'Rrs_645',
        667: 'Rrs_667',
        678: 'Rrs_678'
    }

    with Dataset(filename, 'r') as ds:
        # Access groups (MODIS structure similar to PACE)
        if 'geophysical_data' in ds.groups:
            gd = ds.groups['geophysical_data']
            nav = ds.groups['navigation_data']
        else:
            # Flat structure
            gd = ds
            nav = ds

        # Load coordinates
        lats = nav.variables['latitude'][:]
        lons = nav.variables['longitude'][:]

        # Load flags
        if 'l2_flags' in gd.variables:
            flags = gd.variables['l2_flags'][:]
        else:
            flags = np.zeros(lats.shape, dtype=np.int32)

        # Find and load requested wavelengths
        data_list = []
        actual_wls = []

        for target_wl in wavelengths:
            # Find closest MODIS band
            available = np.array(list(modis_bands.keys()))
            idx = np.argmin(np.abs(available - target_wl))
            closest_wl = available[idx]
            var_name = modis_bands[closest_wl]

            if var_name in gd.variables:
                actual_wls.append(closest_wl)
                data_list.append(gd.variables[var_name][:])
            else:
                print(f"Warning: {var_name} not found in file, skipping wavelength {target_wl}")

        if len(data_list) == 0:
            raise ValueError("No valid wavelength channels found in MODIS file")

        data = np.array(data_list)  # Shape: (n_wavelengths, nx, ny)
        actual_wavelengths = np.array(actual_wls)

    return data, lats, lons, flags, actual_wavelengths


def create_scene_plot(data: np.ndarray, lats: np.ndarray, lons: np.ndarray,
                      flags: np.ndarray, wavelengths: np.ndarray,
                      filename: str, output_html: Optional[str] = None,
                      ncols: int = 3) -> None:
    """
    Create a multi-panel Bokeh plot of ocean color scene.

    Parameters
    ----------
    data : np.ndarray
        Data array of shape (n_wavelengths, nx, ny)
    lats : np.ndarray
        Latitude array of shape (nx, ny)
    lons : np.ndarray
        Longitude array of shape (nx, ny)
    flags : np.ndarray
        Flag array of shape (nx, ny)
    wavelengths : np.ndarray
        Wavelength values for each channel
    filename : str
        Source filename for title
    output_html : str, optional
        Path to save HTML output
    ncols : int
        Number of columns in grid layout
    """
    n_wavelengths = len(wavelengths)

    # Set output file if specified
    if output_html:
        output_file(output_html, title="Ocean Color Scene")

    # Create figure grid
    figures = []

    # Determine grid dimensions
    nrows = int(np.ceil(n_wavelengths / ncols))

    for i, wl in enumerate(wavelengths):
        channel_data = data[i, :, :]

        # Mask invalid pixels (set to NaN for proper display)
        masked_data = channel_data.copy()
        masked_data[flags != 0] = np.nan
        masked_data[~np.isfinite(channel_data)] = np.nan

        # Compute valid data range for color mapping
        valid_mask = np.isfinite(masked_data)
        if np.any(valid_mask):
            vmin = np.nanpercentile(masked_data[valid_mask], 1)
            vmax = np.nanpercentile(masked_data[valid_mask], 99)
        else:
            vmin, vmax = 0, 1

        # Ensure non-zero range
        if vmax == vmin:
            vmax = vmin + 1e-6

        # Create color mapper
        color_mapper = LinearColorMapper(
            palette=Viridis256,
            low=vmin,
            high=vmax,
            nan_color='gray'
        )

        # Create figure
        p = figure(
            width=400,
            height=350,
            title=f"{wl:.1f} nm",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            match_aspect=True,
            aspect_ratio=1.0
        )

        # Plot data as image
        p.image(
            image=[masked_data],
            x=0,
            y=0,
            dw=masked_data.shape[1],
            dh=masked_data.shape[0],
            color_mapper=color_mapper,
        )

        # Add color bar
        color_bar = ColorBar(
            color_mapper=color_mapper,
            ticker=BasicTicker(desired_num_ticks=5),
            width=10,
            location=(0, 0),
            title="Rrs" if "Rrs" in str(type(data)) else "Value"
        )
        p.add_layout(color_bar, 'right')

        # Add hover tool with lat/lon
        # Note: For hover to show lat/lon at cursor position, we'd need to
        # convert to a different format. Here we keep it simple.
        hover = HoverTool(
            tooltips=[
                ("Wavelength", f"{wl:.1f} nm"),
                ("x, y", "$x{int}, $y{int}"),
                ("Value", "@image{0.0000}"),
            ]
        )
        p.add_tools(hover)

        # Style
        p.xaxis.axis_label = "X (pixels)"
        p.yaxis.axis_label = "Y (pixels)"
        p.title.text_font_size = "12pt"

        figures.append(p)

    # Fill remaining grid positions with None for even layout
    while len(figures) % ncols != 0:
        figures.append(None)

    # Create grid layout
    grid = []
    for i in range(0, len(figures), ncols):
        grid.append(figures[i:i+ncols])

    # Create overall title
    print(f"\nDisplaying {n_wavelengths} wavelength channels from {os.path.basename(filename)}")
    print(f"Wavelengths: {wavelengths}")

    # Show plot
    layout = gridplot(grid, sizing_mode='scale_width')
    show(layout)

    print(f"\nPlot displayed in browser")
    if output_html:
        print(f"HTML saved to: {output_html}")


def parser() -> argparse.ArgumentParser:
    """Build and return argument parser."""
    p = argparse.ArgumentParser(
        description="Plot ocean color granule as multi-panel scene",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s PACE_OCI.L2.nc
  %(prog)s PACE_OCI.L2.nc --channels 400,450,550,650
  %(prog)s PACE_OCI.L1C.nc --channels 400,500,600,700,800
  %(prog)s granule.nc -o scene.html
        """
    )

    p.add_argument("granule", help="Path to ocean color granule netCDF file")

    p.add_argument(
        "--channels",
        type=str,
        default="350,400,452,552,600,650,700,750,801",
        help="Comma-separated wavelengths to plot (nm). Default: 350,400,452,552,600,650,700,750,801"
    )

    p.add_argument(
        "--ncols",
        type=int,
        default=3,
        help="Number of columns in grid layout. Default: 3"
    )

    p.add_argument(
        "-o", "--output",
        help="Output HTML file path (optional)"
    )

    return p


def main(args: argparse.Namespace = None):
    """Main entry point."""
    if args is None:
        p = parser()
        args = p.parse_args()

    # Parse wavelengths
    try:
        wavelengths = [float(w.strip()) for w in args.channels.split(',')]
    except ValueError:
        print("Error: Invalid wavelength format. Use comma-separated numbers.")
        sys.exit(1)

    # Check file exists
    if not os.path.exists(args.granule):
        print(f"Error: File not found: {args.granule}")
        sys.exit(1)

    # Detect file type
    print(f"Loading: {args.granule}")
    file_type = detect_file_type(args.granule)
    print(f"Detected file type: {file_type}")

    # Load data based on file type
    try:
        if file_type == 'PACE_L2':
            data, lats, lons, flags, actual_wls = load_pace_l2_scene(args.granule, wavelengths)
        elif file_type == 'PACE_L1C':
            data, lats, lons, flags, actual_wls = load_pace_l1c_scene(args.granule, wavelengths)
        elif file_type == 'MODIS_L2':
            data, lats, lons, flags, actual_wls = load_modis_l2_scene(args.granule, wavelengths)
        else:
            print(f"Error: Unsupported file type: {file_type}")
            sys.exit(1)
    except Exception as e:
        print(f"Error loading file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Create plot
    create_scene_plot(
        data, lats, lons, flags, actual_wls,
        args.granule,
        output_html=args.output,
        ncols=args.ncols
    )


if __name__ == "__main__":
    main()
