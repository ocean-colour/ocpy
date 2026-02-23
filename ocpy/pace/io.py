""" I/O for PACE """

import numpy as np
import xarray as xr
from netCDF4 import Dataset

from IPython import embed

def load_oci_l2(fn, full_flag:bool=False):
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

    # create the initial dataset so that we have all the attributes
    xds = xr.open_dataset(fn)
    # open the file with netCDF to get all the actual data
    dataset = Dataset(fn)
    
    # grab the necessary group data
    gd    = dataset.groups['geophysical_data']
    nav   = dataset.groups['navigation_data']
    lons  = nav.variables["longitude"][:]
    lats  = nav.variables["latitude"][:]
    flags = gd.variables["l2_flags"][:]
    wls = dataset.groups['sensor_band_parameters']['wavelength_3d'][:].data

    # create the dataset, now we're only adding in Rrs, Rrs_unc
    # but the options are: ['Rrs', 'Rrs_unc', 'aot_865', 'angstrom', 'avw', 'l2_flags']
    #embed(header='41 of io.py')
    rrs_xds = xr.Dataset(
        {'Rrs':(('x', 'y', 'wl'),gd.variables['Rrs'][:].data)},
               coords = {'latitude': (('x', 'y'), lats),
                                  'longitude': (('x', 'y'), lons),
                                  'wavelength' : ('wl', wls)},
               attrs={'variable':'Remote sensing reflectance'})
    rrsu_xds = xr.Dataset(
        {'Rrs_unc':(('x', 'y', 'wl'),gd.variables['Rrs_unc'][:].data)},
               coords = {'latitude': (('x', 'y'), lats),
                                  'longitude': (('x', 'y'), lons),
                                  'wavelength' : ('wl', wls)},
               attrs={'variable':'Remote sensing reflectance error'})
    nflh_xds = xr.Dataset(
        {'nflh':(('x', 'y'),gd.variables['nflh'][:].data)},
               coords = {'latitude': (('x', 'y'), lats),
                                  'longitude': (('x', 'y'), lons)},
               attrs={'variable':'Normalized Fluorescence Line Height'})
    

    # merge back into the xarray dataset with all the attributes
    xds['Rrs'] = rrs_xds.Rrs
    xds['Rrs_unc'] = rrsu_xds.Rrs_unc
    xds['FLH'] = nflh_xds.nflh

    # replace nodata areas with nan
    #xds = xds.where(xds['Rrs'] != -32767.0)

    # eliminate everything that isn't a flag bit of 0 (meaning no flags)
    if full_flag:
        xds['Rrs'] = xr.where(xr.DataArray(flags.data, dims=['x', 'y'])==0, xds['Rrs'], np.nan)
        
    return xds, flags


def load_iop_l2(fn:str):
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
    # create the initial dataset so that we have all the attributes
    xds = xr.open_dataset(fn)
    # open the file with netCDF to get all the actual data
    dataset = Dataset(fn)
    
    # grab the necessary group data
    gd    = dataset.groups['geophysical_data']
    nav   = dataset.groups['navigation_data']
    lons  = nav.variables["longitude"][:]
    lats  = nav.variables["latitude"][:]
    flags = gd.variables["l2_flags"][:]
    wls = dataset.groups['sensor_band_parameters']['wavelength_3d'][:].data

    #embed(header='95 of io.py')
    rrs_xds = xr.Dataset(
        {'a':(('x', 'y', 'wl'),gd.variables['a'][:].data)},
               coords = {'latitude': (('x', 'y'), lats),
                                  'longitude': (('x', 'y'), lons),
                                  'wavelength' : ('wl', wls)},
               attrs={'variable':'Total absorption coefficient'})
    rrsu_xds = xr.Dataset(
        {'bb':(('x', 'y', 'wl'),gd.variables['bb'][:].data)},
               coords = {'latitude': (('x', 'y'), lats),
                                  'longitude': (('x', 'y'), lons),
                                  'wavelength' : ('wl', wls)},
               attrs={'variable':'Total backscatter coefficient'})
    aph_xds = xr.Dataset(
        {'aph':(('x', 'y', 'wl'),gd.variables['aph'][:].data)},
               coords = {'latitude': (('x', 'y'), lats),
                                  'longitude': (('x', 'y'), lons),
                                  'wavelength' : ('wl', wls)},
               attrs={'variable':'Phytoplankton absorption spectrum'})
    adgs_xds = xr.Dataset(
        {'adg_s':(('x', 'y'),gd.variables['adg_s'][:].data)},
               coords = {'latitude': (('x', 'y'), lats),
                                  'longitude': (('x', 'y'), lons)},
               attrs={'variable':'adg spectral parameter'})
    adg_xds = xr.Dataset(
        {'adg_442':(('x', 'y'),gd.variables['adg_442'][:].data)},
               coords = {'latitude': (('x', 'y'), lats),
                                  'longitude': (('x', 'y'), lons)},
               attrs={'variable':'adg 442 value'})
    bbp442_xds = xr.Dataset(
        {'bbp_442':(('x', 'y'),gd.variables['bbp_442'][:].data)},
               coords = {'latitude': (('x', 'y'), lats),
                                  'longitude': (('x', 'y'), lons)},
               attrs={'variable':'bbp 442 value'})
    bbpunc442_xds = xr.Dataset(
        {'bbp_unc_442':(('x', 'y'),gd.variables['bbp_unc_442'][:].data)},
               coords = {'latitude': (('x', 'y'), lats),
                                  'longitude': (('x', 'y'), lons)},
               attrs={'variable':'bbp 442 uncertainty'})
    bbps_xds = xr.Dataset(
        {'bbp_s':(('x', 'y'),gd.variables['bbp_s'][:].data)},
               coords = {'latitude': (('x', 'y'), lats),
                                  'longitude': (('x', 'y'), lons)},
               attrs={'variable':'bbp spectral shape value'})
    

    # merge back into the xarray dataset with all the attributes
    xds['a'] = rrs_xds.a
    xds['bb'] = rrsu_xds.bb
    xds['aph'] = aph_xds.aph
    xds['adg_s'] = adgs_xds.adg_s
    xds['adg_442'] = adg_xds.adg_442
    xds['bbp_442'] = bbp442_xds.bbp_442
    xds['bbp_unc_442'] = bbpunc442_xds.bbp_unc_442
    xds['bbp_s'] = bbps_xds.bbp_s

    # replace nodata areas with nan
    #xds = xds.where(xds['Rrs'] != -32767.0)

    # eliminate everything that isn't a flag bit of 0 (meaning no flags)
    #if full_flag:
    #    xds['Rrs'] = xr.where(xr.DataArray(flags.data, dims=['x', 'y'])==0, xds['Rrs'], np.nan)
        
    return xds, flags