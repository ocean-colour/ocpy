======================
Satellite Data Processing
======================

This guide covers working with ocean color satellite data using ocpy.

Introduction
------------

Ocean color satellites provide global observations of marine ecosystems. ocpy supports:

* **PACE**: NASA's newest hyperspectral mission
* **MODIS**: Long-term climate record (Aqua/Terra)
* **SeaWiFS**: Historical reference (1997-2010)
* **VIIRS**: Current operational mission

Working with PACE Data
----------------------

PACE (Plankton, Aerosol, Cloud, ocean Ecosystem) provides hyperspectral ocean color
with unprecedented spectral resolution.

Wavelength Configuration
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from oceancolor.satellites import pace

   # Generate PACE OCI wavelength array
   wavelengths = pace.wave(wv_min=400, wv_max=700, step=5)
   print(f"PACE wavelengths: {wavelengths}")
   print(f"Number of bands: {len(wavelengths)}")

   # Full hyperspectral coverage
   wavelengths_full = pace.wave(wv_min=340, wv_max=890, step=5)

Noise Model
^^^^^^^^^^^

Understanding PACE uncertainties:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from oceancolor.satellites import pace

   wavelengths = pace.wave(wv_min=400, wv_max=700, step=5)

   # Generate noise vector
   noise = pace.gen_noise_vector(wavelengths, include_sampling=True)

   # Plot noise vs wavelength
   plt.figure(figsize=(10, 6))
   plt.plot(wavelengths, noise * 100)  # Convert to percent
   plt.xlabel('Wavelength (nm)')
   plt.ylabel('Noise (%)')
   plt.title('PACE OCI Noise Model')
   plt.grid(True, alpha=0.3)

Loading PACE Data
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from oceancolor.pace import io as pace_io

   # Load Level 2 OC data
   Rrs, Rrs_unc, FLH = pace_io.load_oci_l2('PACE_OCI_L2.nc')

   print(f"Rrs dimensions: {Rrs.dims}")
   print(f"Wavelengths: {Rrs.wavelength.values[:10]}...")  # First 10

   # Load IOP data
   iop_data = pace_io.load_iop_l2('PACE_OCI_L2_IOP.nc')

Working with MODIS Data
-----------------------

Matchup Datasets
^^^^^^^^^^^^^^^^

.. code-block:: python

   from oceancolor.satellites import modis

   # Load MODIS matchup dataset
   matchups = modis.load_matchups()

   print(f"Number of matchups: {len(matchups)}")
   print(f"Columns: {matchups.columns.tolist()}")

   # Calculate error statistics
   errors = modis.calc_errors(rel_in_situ_error=0.05)
   print(f"Error statistics:\n{errors}")

MODIS Wavelengths
^^^^^^^^^^^^^^^^^

.. code-block:: python

   # MODIS ocean color bands
   modis_bands = {
       8: 412,   # Dissolved organic matter
       9: 443,   # Chlorophyll absorption (blue)
       10: 488,  # Chlorophyll absorption
       11: 531,  # Chlorophyll
       12: 551,  # Green reference
       13: 667,  # Atmospheric correction
       14: 678,  # Chlorophyll fluorescence
       15: 748,  # Aerosol, atmospheric correction
       16: 869,  # Aerosol
   }

Working with SeaWiFS Data
-------------------------

.. code-block:: python

   from oceancolor.satellites import seawifs

   # Load matchup data
   matchups = seawifs.load_matchups()

   # Calculate errors
   errors = seawifs.calc_errors(rel_in_situ_error=0.05)

   # SeaWiFS wavelengths
   seawifs_bands = [412, 443, 490, 510, 555, 670, 765, 865]

Cross-Sensor Comparison
-----------------------

Comparing data from different sensors requires careful band adjustment:

.. code-block:: python

   import numpy as np
   from scipy.interpolate import interp1d

   def band_shift(Rrs, wave_source, wave_target):
       """Shift Rrs from source to target wavelengths."""
       f = interp1d(wave_source, Rrs, kind='linear',
                    fill_value='extrapolate')
       return f(wave_target)

   # Example: MODIS to SeaWiFS comparison
   modis_wave = np.array([412, 443, 488, 531, 551, 667])
   seawifs_wave = np.array([412, 443, 490, 510, 555, 670])

   # Shift MODIS Rrs to SeaWiFS bands
   Rrs_modis = np.array([0.003, 0.004, 0.005, 0.006, 0.007, 0.001])
   Rrs_modis_shifted = band_shift(Rrs_modis, modis_wave, seawifs_wave)

   print(f"Original MODIS Rrs: {Rrs_modis}")
   print(f"Shifted to SeaWiFS bands: {Rrs_modis_shifted}")

Quality Control
---------------

Satellite data requires careful quality control:

.. code-block:: python

   import numpy as np

   def apply_l2_flags(data, flags, bad_bits=None):
       """Apply Level 2 quality flags.

       Parameters
       ----------
       data : ndarray
           Data array to mask
       flags : ndarray
           Flag array (same shape as data spatial dims)
       bad_bits : list, optional
           List of flag bits to mask (default: standard ocean color flags)

       Returns
       -------
       data_masked : ndarray
           Masked data array
       """
       if bad_bits is None:
           # Standard bad flags for ocean color
           bad_bits = [
               0,   # ATMFAIL - Atmospheric correction failure
               1,   # LAND - Land pixel
               3,   # HIGLINT - High sun glint
               4,   # HILT - High sensor tilt
               7,   # STRAYLIGHT - Stray light
               8,   # CLDICE - Cloud or ice
               12,  # LOWLW - Low water-leaving radiance
           ]

       # Create mask
       mask = np.zeros(flags.shape, dtype=bool)
       for bit in bad_bits:
           mask |= (flags & (1 << bit)) > 0

       # Apply mask
       if data.ndim > flags.ndim:
           # Data has spectral dimension
           mask = np.broadcast_to(mask[..., np.newaxis], data.shape)

       data_masked = np.where(mask, np.nan, data)
       return data_masked

   # Example usage
   # Rrs_clean = apply_l2_flags(Rrs, l2_flags)

Atmospheric Correction Validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   def check_atm_correction(Rrs, wavelengths):
       """Check for atmospheric correction issues."""
       issues = []

       # Find NIR band
       idx_nir = wavelengths > 700
       if np.any(idx_nir):
           Rrs_nir = np.nanmean(Rrs[..., idx_nir], axis=-1)

           # NIR should be near zero over ocean
           if np.nanmean(Rrs_nir) > 0.001:
               issues.append('High NIR reflectance - possible cloud/aerosol')

       # Check for negative blue
       idx_blue = wavelengths < 450
       if np.any(idx_blue):
           Rrs_blue = Rrs[..., idx_blue]
           if np.any(Rrs_blue < -0.001):
               issues.append('Negative blue Rrs - atmospheric overcorrection')

       # Check spectral shape
       idx_443 = np.argmin(np.abs(wavelengths - 443))
       idx_555 = np.argmin(np.abs(wavelengths - 555))

       ratio = Rrs[..., idx_443] / Rrs[..., idx_555]
       if np.nanmean(ratio) > 2.0:
           issues.append('Unusual blue/green ratio')

       return issues

Creating Time Series
--------------------

Building climate-quality time series:

.. code-block:: python

   import numpy as np
   import pandas as pd

   def create_timeseries(files, region_bounds, product='chlor_a'):
       """Create time series from multiple satellite files.

       Parameters
       ----------
       files : list
           List of satellite file paths
       region_bounds : dict
           Region bounds: {'lat_min', 'lat_max', 'lon_min', 'lon_max'}
       product : str
           Product name to extract

       Returns
       -------
       ts : pd.DataFrame
           Time series DataFrame
       """
       timestamps = []
       values = []

       for f in files:
           # Load file (sensor-dependent)
           # This is pseudocode - adapt for your file format
           # data = load_satellite_file(f)

           # Extract timestamp from filename or metadata
           # timestamp = parse_timestamp(f)

           # Subset to region
           # region_data = subset_region(data, region_bounds)

           # Calculate regional mean
           # regional_mean = np.nanmean(region_data[product])

           # timestamps.append(timestamp)
           # values.append(regional_mean)
           pass

       ts = pd.DataFrame({
           'time': timestamps,
           product: values
       })
       ts.set_index('time', inplace=True)

       return ts

Plotting Satellite Data
-----------------------

Spatial Maps
^^^^^^^^^^^^

.. code-block:: python

   import matplotlib.pyplot as plt
   import matplotlib.colors as colors
   import numpy as np

   def plot_rrs_map(Rrs, lat, lon, wavelength_idx, vmin=0, vmax=0.02):
       """Plot Rrs map at a specific wavelength."""
       fig, ax = plt.subplots(figsize=(12, 8))

       im = ax.pcolormesh(lon, lat, Rrs[..., wavelength_idx],
                          vmin=vmin, vmax=vmax,
                          cmap='viridis', shading='auto')

       cbar = plt.colorbar(im, ax=ax)
       cbar.set_label('Rrs (sr⁻¹)')

       ax.set_xlabel('Longitude')
       ax.set_ylabel('Latitude')

       return fig, ax

   def plot_chl_map(chl, lat, lon):
       """Plot chlorophyll map with log scale."""
       fig, ax = plt.subplots(figsize=(12, 8))

       norm = colors.LogNorm(vmin=0.01, vmax=30)
       im = ax.pcolormesh(lon, lat, chl,
                          norm=norm,
                          cmap='viridis', shading='auto')

       cbar = plt.colorbar(im, ax=ax, extend='both')
       cbar.set_label('Chlorophyll-a (mg/m³)')

       ax.set_xlabel('Longitude')
       ax.set_ylabel('Latitude')

       return fig, ax

Spectra Plots
^^^^^^^^^^^^^

.. code-block:: python

   def plot_pixel_spectrum(Rrs, Rrs_unc, wavelengths, lat, lon):
       """Plot Rrs spectrum with uncertainty."""
       fig, ax = plt.subplots(figsize=(10, 6))

       ax.fill_between(wavelengths,
                       Rrs - Rrs_unc, Rrs + Rrs_unc,
                       alpha=0.3, label='Uncertainty')
       ax.plot(wavelengths, Rrs, 'b-', linewidth=2, label='Rrs')

       ax.set_xlabel('Wavelength (nm)')
       ax.set_ylabel('Rrs (sr⁻¹)')
       ax.set_title(f'Spectrum at ({lat:.2f}, {lon:.2f})')
       ax.legend()
       ax.grid(True, alpha=0.3)

       return fig, ax

File Formats
------------

Common satellite file formats:

* **NetCDF (.nc)**: Standard format for Level 2 and Level 3 products
* **HDF5 (.h5)**: Used by some missions
* **GeoTIFF**: For mapped products

.. code-block:: python

   import xarray as xr
   import netCDF4 as nc

   # Reading with xarray (recommended)
   def read_l2_xarray(filepath):
       """Read L2 file with xarray."""
       ds = xr.open_dataset(filepath)
       return ds

   # Reading with netCDF4
   def read_l2_netcdf(filepath, variables=None):
       """Read specific variables from L2 file."""
       with nc.Dataset(filepath, 'r') as f:
           if variables is None:
               variables = list(f.variables.keys())

           data = {}
           for var in variables:
               if var in f.variables:
                   data[var] = f.variables[var][:]

           return data

References
----------

* NASA Ocean Biology Processing Group: https://oceancolor.gsfc.nasa.gov/

* Werdell, P.J., et al. (2019). The Plankton, Aerosol, Cloud, ocean Ecosystem
  mission. BAMS, 100(9), 1775-1794.

* McClain, C.R., et al. (2004). An overview of the SeaWiFS project. Deep Sea
  Research Part II, 51(1-3), 5-42.
