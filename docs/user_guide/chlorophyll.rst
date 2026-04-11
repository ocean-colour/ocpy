=======================
Chlorophyll Estimation
=======================

This guide covers estimating chlorophyll-a concentration from ocean color data using ocpy.

Introduction
------------

Chlorophyll-a (Chl-a) is the primary pigment in phytoplankton and serves as a proxy
for phytoplankton biomass. It's one of the most important products from ocean color
remote sensing.

ocpy provides standard band-ratio algorithms (OC2, OC4) that relate spectral ratios
of Rrs to chlorophyll concentration.

Band Ratio Algorithms
---------------------

OC4 Algorithm
^^^^^^^^^^^^^

The OC4 algorithm uses the maximum of three blue/green band ratios:

.. code-block:: python

   import numpy as np
   from oceancolor.chl.band_ratios import oc4

   # Standard wavelengths
   wavelengths = np.array([443, 490, 510, 555, 670])

   # Example Rrs spectrum
   Rrs = np.array([0.0045, 0.0052, 0.0055, 0.0060, 0.0012])

   # Calculate chlorophyll
   chl = oc4(wavelengths, Rrs)
   print(f"Chlorophyll-a: {chl:.2f} mg/m³")

The algorithm:

1. Computes three ratios: Rrs(443)/Rrs(555), Rrs(490)/Rrs(555), Rrs(510)/Rrs(555)
2. Takes the maximum ratio
3. Applies a 4th-order polynomial in log-space

OC2 Algorithm
^^^^^^^^^^^^^

The simpler OC2 algorithm uses only the 490/555 ratio:

.. code-block:: python

   from oceancolor.chl.band_ratios import oc2

   chl = oc2(wavelengths, Rrs)
   print(f"Chlorophyll-a (OC2): {chl:.2f} mg/m³")

Processing Satellite Data
-------------------------

Here's how to apply the algorithms to satellite imagery:

.. code-block:: python

   import numpy as np
   from oceancolor.chl.band_ratios import oc4

   def process_chlorophyll(Rrs_cube, wavelengths, flags=None):
       """Calculate chlorophyll for a satellite image.

       Parameters
       ----------
       Rrs_cube : ndarray
           3D array (rows, cols, wavelengths)
       wavelengths : ndarray
           Array of wavelengths in nm
       flags : ndarray, optional
           2D quality flag array

       Returns
       -------
       chl : ndarray
           2D chlorophyll array (mg/m³)
       """
       nrows, ncols = Rrs_cube.shape[:2]
       chl = np.full((nrows, ncols), np.nan)

       for i in range(nrows):
           for j in range(ncols):
               # Skip flagged pixels
               if flags is not None and flags[i, j] != 0:
                   continue

               Rrs = Rrs_cube[i, j, :]

               # Skip invalid spectra
               if np.any(np.isnan(Rrs)) or np.any(Rrs < 0):
                   continue

               try:
                   chl[i, j] = oc4(wavelengths, Rrs)
               except Exception:
                   continue

       return chl

   # Example usage
   # chl_map = process_chlorophyll(Rrs_data, wavelengths, quality_flags)

Handling Different Sensors
--------------------------

Different satellites have different band centers. Here's how to handle this:

.. code-block:: python

   from scipy.interpolate import interp1d
   import numpy as np

   # Standard OC4 wavelengths
   oc4_wavelengths = np.array([443, 490, 510, 555, 670])

   def chl_from_sensor(sensor_wavelengths, sensor_Rrs, sensor_name='generic'):
       """Calculate chlorophyll from any sensor.

       Interpolates to standard OC4 wavelengths.
       """
       from oceancolor.chl.band_ratios import oc4

       # Sensor-specific wavelength mappings
       sensor_bands = {
           'modis': np.array([443, 488, 531, 551, 667]),
           'viirs': np.array([443, 486, 551, 551, 671]),
           'seawifs': np.array([443, 490, 510, 555, 670]),
           'pace': np.array([443, 490, 510, 555, 670]),
       }

       if sensor_name.lower() in sensor_bands:
           source_waves = sensor_bands[sensor_name.lower()]
       else:
           source_waves = sensor_wavelengths

       # Interpolate to standard wavelengths
       f = interp1d(source_waves, sensor_Rrs, kind='linear',
                    fill_value='extrapolate')
       Rrs_standard = f(oc4_wavelengths)

       return oc4(oc4_wavelengths, Rrs_standard)

   # Example for MODIS
   modis_wavelengths = np.array([443, 488, 531, 551, 667])
   modis_Rrs = np.array([0.0045, 0.0052, 0.0058, 0.0060, 0.0012])

   chl = chl_from_sensor(modis_wavelengths, modis_Rrs, sensor_name='modis')
   print(f"MODIS Chlorophyll: {chl:.2f} mg/m³")

Quality Control
---------------

Chlorophyll estimates need quality control:

.. code-block:: python

   import numpy as np

   def qc_chlorophyll(chl, Rrs, wavelengths):
       """Quality control for chlorophyll estimates.

       Returns
       -------
       chl_qc : ndarray
           Chlorophyll with QC applied (bad values set to NaN)
       flags : ndarray
           Quality flags
       """
       chl_qc = chl.copy()
       flags = np.zeros_like(chl, dtype=int)

       # Flag 1: Very low chlorophyll (below algorithm range)
       low_chl = chl < 0.01
       flags[low_chl] |= 1
       chl_qc[low_chl] = np.nan

       # Flag 2: Very high chlorophyll (above algorithm range)
       high_chl = chl > 100
       flags[high_chl] |= 2
       chl_qc[high_chl] = np.nan

       # Flag 4: Negative Rrs (atmospheric correction failure)
       idx_blue = wavelengths < 500
       bad_atm = np.any(Rrs[..., idx_blue] < 0, axis=-1)
       flags[bad_atm] |= 4
       chl_qc[bad_atm] = np.nan

       # Flag 8: Red > green (sediment-dominated)
       idx_green = np.argmin(np.abs(wavelengths - 555))
       idx_red = np.argmin(np.abs(wavelengths - 670))
       sediment = Rrs[..., idx_red] > Rrs[..., idx_green]
       flags[sediment] |= 8
       # Don't set to NaN, but flag it

       return chl_qc, flags

Algorithm Limitations
---------------------

Band-ratio algorithms have known limitations:

1. **Optically Complex Waters**

   In coastal waters with high CDOM or sediments, band-ratio algorithms
   can overestimate chlorophyll:

   .. code-block:: python

      # Check for optically complex conditions
      def is_complex_water(Rrs, wavelengths):
          """Detect optically complex waters."""
          idx_443 = np.argmin(np.abs(wavelengths - 443))
          idx_555 = np.argmin(np.abs(wavelengths - 555))
          idx_670 = np.argmin(np.abs(wavelengths - 670))

          # High CDOM: very low blue reflectance
          high_cdom = Rrs[idx_443] < 0.001

          # High sediments: high red reflectance
          high_sediment = Rrs[idx_670] > 0.002

          return high_cdom or high_sediment

2. **Algorithm Saturation**

   Above ~30 mg/m³, band ratios saturate:

   .. code-block:: python

      # Maximum reliable chlorophyll
      CHL_MAX_RELIABLE = 30.0

      chl_flag = chl > CHL_MAX_RELIABLE

3. **Very Low Chlorophyll**

   Below ~0.05 mg/m³, uncertainties increase significantly.

Alternative Approaches
----------------------

For challenging conditions, consider:

1. **Red/NIR algorithms** for turbid waters
2. **Semi-analytical algorithms** (LS2, QAA) for IOP-based chlorophyll
3. **Algorithm blending** across different water types

.. code-block:: python

   def blended_chlorophyll(Rrs, wavelengths, water_type='auto'):
       """Blended chlorophyll algorithm.

       Uses different algorithms based on water type.
       """
       from oceancolor.chl.band_ratios import oc4
       from oceancolor.ls2 import ls2_main

       if water_type == 'auto':
           # Classify water type
           if is_complex_water(Rrs, wavelengths):
               water_type = 'complex'
           else:
               water_type = 'case1'

       if water_type == 'case1':
           # Standard band-ratio for open ocean
           return oc4(wavelengths, Rrs)

       elif water_type == 'complex':
           # Use IOP-based approach for complex waters
           # This is a simplified example
           # In practice, derive a_ph and convert to Chl
           pass

       return np.nan

Uncertainty Estimation
----------------------

Estimate chlorophyll uncertainties:

.. code-block:: python

   import numpy as np

   def chl_uncertainty(chl, Rrs_unc, wavelengths):
       """Estimate chlorophyll uncertainty from Rrs uncertainty.

       Uses error propagation through the OC4 algorithm.
       """
       from oceancolor.chl.band_ratios import oc4

       # Monte Carlo approach
       n_samples = 100
       chl_samples = []

       for _ in range(n_samples):
           # Perturb Rrs within uncertainties
           Rrs_perturbed = Rrs + np.random.normal(0, Rrs_unc)
           Rrs_perturbed = np.maximum(Rrs_perturbed, 1e-6)  # Keep positive

           try:
               chl_sample = oc4(wavelengths, Rrs_perturbed)
               chl_samples.append(chl_sample)
           except:
               continue

       if len(chl_samples) > 10:
           chl_std = np.std(chl_samples)
           chl_cv = chl_std / chl * 100  # Coefficient of variation (%)
           return chl_std, chl_cv
       else:
           return np.nan, np.nan

Visualization
-------------

Plotting chlorophyll maps:

.. code-block:: python

   import matplotlib.pyplot as plt
   import matplotlib.colors as mcolors
   import numpy as np

   def plot_chlorophyll_map(chl, lat, lon, title='Chlorophyll-a'):
       """Plot chlorophyll concentration map."""
       fig, ax = plt.subplots(figsize=(12, 8))

       # Log-scale colormap for chlorophyll
       norm = mcolors.LogNorm(vmin=0.01, vmax=30)

       im = ax.pcolormesh(lon, lat, chl, norm=norm,
                          cmap='viridis', shading='auto')

       cbar = plt.colorbar(im, ax=ax, extend='both')
       cbar.set_label('Chlorophyll-a (mg/m³)')

       ax.set_xlabel('Longitude')
       ax.set_ylabel('Latitude')
       ax.set_title(title)

       return fig, ax

   # For spatial plots with cartopy
   def plot_chl_with_coast(chl, lat, lon):
       """Plot chlorophyll with coastlines."""
       import cartopy.crs as ccrs
       import cartopy.feature as cfeature

       fig, ax = plt.subplots(
           figsize=(12, 8),
           subplot_kw={'projection': ccrs.PlateCarree()}
       )

       norm = mcolors.LogNorm(vmin=0.01, vmax=30)
       im = ax.pcolormesh(lon, lat, chl, norm=norm,
                          transform=ccrs.PlateCarree(),
                          cmap='viridis', shading='auto')

       ax.add_feature(cfeature.COASTLINE)
       ax.add_feature(cfeature.LAND, facecolor='lightgray')
       ax.gridlines(draw_labels=True)

       plt.colorbar(im, ax=ax, label='Chlorophyll-a (mg/m³)')

       return fig, ax

References
----------

* O'Reilly, J.E., et al. (1998). Ocean color chlorophyll algorithms for SeaWiFS.
  JGR, 103(C11), 24937-24953.

* O'Reilly, J.E. and Werdell, P.J. (2019). Chlorophyll algorithms for ocean color
  sensors - OC4, OC5 & OC6. Remote Sensing of Environment, 229, 32-47.

* Sathyendranath, S., et al. (2019). An Ocean-Colour Time Series for Use in
  Climate Studies: The Experience of the Ocean-Colour Climate Change Initiative.
  Sensors, 19(19), 4285.
