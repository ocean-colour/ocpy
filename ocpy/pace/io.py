""" I/O for PACE """

import numpy as np
import xarray as xr
from netCDF4 import Dataset


def load_oci_l2(fn, full_flag: bool = False):
    """
    Load OCI L2 data from a netCDF file.

    Reflectance data is loaded as 'Rrs' and 'Rrs_unc'.

    Parameters:
    - fn (str): The file path of the netCDF file.
    - full_flag (bool): Flag indicating whether to include only data with no flags (default: False).

    Returns:
    - xds (xarray.Dataset): The loaded dataset containing OCI L2 data.
    - flags (numpy.ndarray): The l2_flags data.

    """
    # Kindly provided by Patrick Gray
    with Dataset(fn) as dataset:
        # Get attributes from root group
        attrs = {k: dataset.getncattr(k) for k in dataset.ncattrs()}

        # Grab group references
        gd = dataset.groups['geophysical_data']
        nav = dataset.groups['navigation_data']
        sbp = dataset.groups['sensor_band_parameters']

        # Load coordinates once
        lons = nav.variables["longitude"][:]
        lats = nav.variables["latitude"][:]
        wls = sbp['wavelength_3d'][:]
        flags = gd.variables["l2_flags"][:]

        # Load data variables
        # Options are: ['Rrs', 'Rrs_unc', 'aot_865', 'angstrom', 'avw', 'l2_flags']
        rrs = gd.variables['Rrs'][:]
        rrs_unc = gd.variables['Rrs_unc'][:]
        nflh = gd.variables['nflh'][:]

    # Build dataset directly with all variables
    xds = xr.Dataset(
        {
            'Rrs': (['x', 'y', 'wl'], rrs),
            'Rrs_unc': (['x', 'y', 'wl'], rrs_unc),
            'FLH': (['x', 'y'], nflh),
        },
        coords={
            'latitude': (['x', 'y'], lats),
            'longitude': (['x', 'y'], lons),
            'wavelength': ('wl', wls),
        },
        attrs=attrs,
    )

    # Eliminate everything that isn't a flag bit of 0 (meaning no flags)
    if full_flag:
        xds['Rrs'] = xds['Rrs'].where(flags == 0)

    return xds, flags


def load_oci_l2_spectrum(fn, target_lat: float, target_lon: float):
    """
    Load a single Rrs spectrum at the nearest pixel to a target lat/lon.

    Much faster than load_oci_l2() when only one spectrum is needed,
    as it avoids loading the full Rrs data cube.

    Parameters:
        fn (str): The file path of the netCDF file.
        target_lat (float): Target latitude in degrees.
        target_lon (float): Target longitude in degrees.

    Returns:
        wls (numpy.ndarray): Wavelengths in nm.
        rrs (numpy.ndarray): Rrs spectrum at the nearest pixel (sr^-1).
        rrs_unc (numpy.ndarray): Rrs uncertainty spectrum (sr^-1).
        flag (int): The l2_flag value at the pixel.
        pixel_coords (tuple): The (ix, iy) pixel indices and actual (lat, lon).
    """
    with Dataset(fn) as dataset:
        nav = dataset.groups['navigation_data']
        gd = dataset.groups['geophysical_data']
        sbp = dataset.groups['sensor_band_parameters']

        # Load only lat/lon arrays (much smaller than Rrs cube)
        lats = nav.variables["latitude"][:]
        lons = nav.variables["longitude"][:]

        # Find nearest pixel using squared distance
        dist = (lats - target_lat)**2 + (lons - target_lon)**2
        ix, iy = np.unravel_index(np.argmin(dist), dist.shape)

        # Load only the single spectrum (slicing happens on disk)
        rrs = gd.variables['Rrs'][ix, iy, :]
        rrs_unc = gd.variables['Rrs_unc'][ix, iy, :]
        wls = sbp['wavelength_3d'][:]
        flag = gd.variables["l2_flags"][ix, iy]

    pixel_coords = (int(ix), int(iy), float(lats[ix, iy]), float(lons[ix, iy]))

    return wls, rrs, rrs_unc, flag, pixel_coords


def load_oci_l2_spectrum_pixel(fn, ix: int, iy: int):
    """
    Load a single Rrs spectrum at a specific pixel index.

    Much faster than load_oci_l2() when only one spectrum is needed,
    as it avoids loading the full Rrs data cube.

    Parameters:
        fn (str): The file path of the netCDF file.
        ix (int): X pixel index (along-track).
        iy (int): Y pixel index (cross-track).

    Returns:
        wls (numpy.ndarray): Wavelengths in nm.
        rrs (numpy.ndarray): Rrs spectrum at the pixel (sr^-1).
        rrs_unc (numpy.ndarray): Rrs uncertainty spectrum (sr^-1).
        flag (int): The l2_flag value at the pixel.
        pixel_coords (tuple): The (ix, iy) pixel indices and actual (lat, lon).
    """
    with Dataset(fn) as dataset:
        nav = dataset.groups['navigation_data']
        gd = dataset.groups['geophysical_data']
        sbp = dataset.groups['sensor_band_parameters']

        # Load only the single spectrum (slicing happens on disk)
        rrs = gd.variables['Rrs'][ix, iy, :]
        rrs_unc = gd.variables['Rrs_unc'][ix, iy, :]
        wls = sbp['wavelength_3d'][:]
        flag = gd.variables["l2_flags"][ix, iy]
        lat = nav.variables["latitude"][ix, iy]
        lon = nav.variables["longitude"][ix, iy]

    pixel_coords = (int(ix), int(iy), float(lat), float(lon))

    return wls, rrs, rrs_unc, flag, pixel_coords


def load_iop_l2(fn: str):
    """
    Load IOP (Inherent Optical Properties) Level 2 data from a netCDF file.

    Parameters:
        fn (str): The file path of the netCDF file.

    Returns:
        xds (xr.Dataset): The xarray dataset containing the loaded data.
            Variables include 'a', 'bb', 'aph', 'adg_s', 'adg_442'.
            Coordinates include 'latitude', 'longitude', 'wavelength'.
        flags (numpy.ndarray): The l2_flags data from the netCDF file.
    """
    with Dataset(fn) as dataset:
        # Get attributes from root group
        attrs = {k: dataset.getncattr(k) for k in dataset.ncattrs()}

        # Grab group references
        gd = dataset.groups['geophysical_data']
        nav = dataset.groups['navigation_data']
        sbp = dataset.groups['sensor_band_parameters']

        # Load coordinates once
        lons = nav.variables["longitude"][:]
        lats = nav.variables["latitude"][:]
        wls = sbp['wavelength_3d'][:]
        flags = gd.variables["l2_flags"][:]

        # Load spectral variables (x, y, wl)
        a = gd.variables['a'][:]
        bb = gd.variables['bb'][:]
        aph = gd.variables['aph'][:]

        # Load scalar variables (x, y)
        adg_s = gd.variables['adg_s'][:]
        adg_442 = gd.variables['adg_442'][:]
        bbp_442 = gd.variables['bbp_442'][:]
        bbp_unc_442 = gd.variables['bbp_unc_442'][:]
        bbp_s = gd.variables['bbp_s'][:]

    # Build dataset directly with all variables
    xds = xr.Dataset(
        {
            'a': (['x', 'y', 'wl'], a),
            'bb': (['x', 'y', 'wl'], bb),
            'aph': (['x', 'y', 'wl'], aph),
            'adg_s': (['x', 'y'], adg_s),
            'adg_442': (['x', 'y'], adg_442),
            'bbp_442': (['x', 'y'], bbp_442),
            'bbp_unc_442': (['x', 'y'], bbp_unc_442),
            'bbp_s': (['x', 'y'], bbp_s),
        },
        coords={
            'latitude': (['x', 'y'], lats),
            'longitude': (['x', 'y'], lons),
            'wavelength': ('wl', wls),
        },
        attrs=attrs,
    )

    return xds, flags