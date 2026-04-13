=========
Utilities
=========

.. module:: oceancolor.utils
   :synopsis: Utility functions

The ``utils`` module provides general utility functions for coordinate handling,
I/O operations, PCA analysis, plotting, and spectral processing.

Coordinate Utilities
--------------------

.. module:: oceancolor.utils.coords
   :synopsis: Geographic coordinate utilities

.. autofunction:: oceancolor.utils.coords.dms_to_decimal

.. autofunction:: oceancolor.utils.coords.parse_dms_string

.. autofunction:: oceancolor.utils.coords.distance_from_latlon

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

   from oceancolor.utils.coords import dms_to_decimal, parse_dms_string, distance_from_latlon
   import numpy as np

   # Convert DMS to decimal degrees
   lat_decimal = dms_to_decimal(35, 45, 30, 'N')
   lon_decimal = dms_to_decimal(122, 30, 15, 'W')
   print(f"Coordinates: {lat_decimal}, {lon_decimal}")

   # Parse DMS string
   lat = parse_dms_string("35°45'30\"N")
   print(f"Parsed latitude: {lat}")

   # Calculate distances from a point
   reference_point = (35.0, -70.0)  # lat, lon
   station_coords = np.array([
       [35.5, -70.2],
       [34.8, -69.5],
       [36.0, -71.0]
   ])
   distances = distance_from_latlon(reference_point, station_coords)
   print(f"Distances (km): {distances}")

I/O Utilities
-------------

.. module:: oceancolor.utils.io
   :synopsis: Input/output utilities

.. autofunction:: oceancolor.utils.io.jsonify

.. autofunction:: oceancolor.utils.io.loadjson

.. autofunction:: oceancolor.utils.io.savejson

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

   from oceancolor.utils.io import loadjson, savejson, jsonify
   import numpy as np

   # Save data to JSON
   data = {
       'wavelengths': [400, 450, 500, 550, 600],
       'absorption': [0.01, 0.02, 0.03, 0.05, 0.10],
       'metadata': {'source': 'in-situ', 'date': '2024-01-15'}
   }
   savejson('optical_data.json', data)

   # Load JSON data
   loaded_data = loadjson('optical_data.json')

   # Convert numpy arrays to JSON-compatible format
   np_data = {
       'array': np.array([1, 2, 3]),
       'matrix': np.random.rand(3, 3)
   }
   json_compatible = jsonify(np_data)

PCA Analysis
------------

.. module:: oceancolor.utils.pca
   :synopsis: Principal Component Analysis utilities

.. autofunction:: oceancolor.utils.pca.fit_normal

.. autofunction:: oceancolor.utils.pca.reconstruct

PCA for Spectral Data
^^^^^^^^^^^^^^^^^^^^^

PCA is useful for dimensionality reduction and pattern recognition in spectral data:

.. code-block:: python

   from oceancolor.utils.pca import fit_normal, reconstruct
   import numpy as np

   # Example spectral dataset (N samples x M wavelengths)
   wavelengths = np.arange(400, 701, 5)
   spectra = np.random.rand(100, len(wavelengths))  # 100 spectra

   # Fit PCA with 5 components
   pca_result = fit_normal(spectra, N=5)

   # Reconstruct a spectrum from PCA components
   test_spectrum = spectra[0]
   reconstructed = reconstruct(pca_result, test_spectrum)

   # Compare original and reconstructed
   import matplotlib.pyplot as plt
   plt.plot(wavelengths, test_spectrum, label='Original')
   plt.plot(wavelengths, reconstructed, '--', label='Reconstructed')
   plt.legend()
   plt.xlabel('Wavelength (nm)')

Plotting Utilities
------------------

.. module:: oceancolor.utils.plotting
   :synopsis: Matplotlib plotting utilities

.. autofunction:: oceancolor.utils.plotting.set_fontsize

.. module:: oceancolor.utils.fig_utils
   :synopsis: Figure utilities

.. autofunction:: oceancolor.utils.fig_utils.set_fontsize

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

   import matplotlib.pyplot as plt
   from oceancolor.utils.plotting import set_fontsize

   fig, ax = plt.subplots(figsize=(10, 6))
   ax.plot([1, 2, 3], [1, 4, 9])
   ax.set_xlabel('X axis')
   ax.set_ylabel('Y axis')
   ax.set_title('Example Plot')

   # Set consistent font size
   set_fontsize(ax, 14)

   plt.tight_layout()

Spectral Utilities
------------------

.. module:: oceancolor.utils.spectra
   :synopsis: Spectral processing utilities

.. autofunction:: oceancolor.utils.spectra.rebin

.. autofunction:: oceancolor.utils.spectra.rebin_to_grid

Spectral Rebinning
^^^^^^^^^^^^^^^^^^

Rebinning is essential when comparing data from different instruments:

.. code-block:: python

   from oceancolor.utils.spectra import rebin, rebin_to_grid
   import numpy as np

   # Original high-resolution spectrum
   wave_hires = np.arange(400, 701, 1)  # 1 nm resolution
   Rrs_hires = np.random.rand(len(wave_hires)) * 0.01

   # Rebin to lower resolution
   wave_lowres = np.arange(400, 701, 5)  # 5 nm resolution
   Rrs_lowres = rebin(wave_hires, Rrs_hires, wave_lowres)

   # Rebin to specific satellite bands
   modis_bands = np.array([412, 443, 488, 531, 551, 667])
   Rrs_modis = rebin_to_grid(wave_hires, Rrs_hires, modis_bands)

Catalog Utilities
-----------------

.. module:: oceancolor.utils.cat_utils
   :synopsis: Catalog and ID matching utilities

.. autofunction:: oceancolor.utils.cat_utils.match_ids

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

   from oceancolor.utils.cat_utils import match_ids
   import numpy as np

   # Match station IDs between two datasets
   dataset1_ids = np.array(['ST001', 'ST002', 'ST003', 'ST004'])
   dataset2_ids = np.array(['ST002', 'ST004', 'ST005'])

   # Find matching indices
   idx1, idx2 = match_ids(dataset1_ids, dataset2_ids)
   print(f"Matching stations: {dataset1_ids[idx1]}")

Common Patterns
---------------

Working with Multiple Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from oceancolor.utils import coords, spectra

   def process_matchup(satellite_data, insitu_data, max_distance_km=10):
       """Match satellite and in-situ data by location."""

       matched_pairs = []

       for i, (sat_lat, sat_lon) in enumerate(satellite_data['coords']):
           # Calculate distances to all in-situ stations
           distances = coords.distance_from_latlon(
               (sat_lat, sat_lon),
               insitu_data['coords']
           )

           # Find nearest within threshold
           nearest_idx = np.argmin(distances)
           if distances[nearest_idx] < max_distance_km:
               # Rebin in-situ to satellite wavelengths
               Rrs_rebinned = spectra.rebin(
                   insitu_data['wavelengths'],
                   insitu_data['Rrs'][nearest_idx],
                   satellite_data['wavelengths']
               )
               matched_pairs.append({
                   'sat_idx': i,
                   'insitu_idx': nearest_idx,
                   'distance_km': distances[nearest_idx],
                   'Rrs_sat': satellite_data['Rrs'][i],
                   'Rrs_insitu': Rrs_rebinned
               })

       return matched_pairs

Unit Conversions
^^^^^^^^^^^^^^^^

.. code-block:: python

   # Common unit conversions in ocean optics

   def Rrs_to_rrs(Rrs):
       """Convert above-water to below-water remote sensing reflectance."""
       # rrs = Rrs / (0.52 + 1.7 * Rrs)
       return Rrs / (0.52 + 1.7 * Rrs)

   def rrs_to_Rrs(rrs):
       """Convert below-water to above-water remote sensing reflectance."""
       # Rrs = 0.52 * rrs / (1 - 1.7 * rrs)
       return 0.52 * rrs / (1 - 1.7 * rrs)

   def nLw_to_Rrs(nLw, F0):
       """Convert normalized water-leaving radiance to Rrs.

       Parameters
       ----------
       nLw : array-like
           Normalized water-leaving radiance (mW/cm²/µm/sr)
       F0 : array-like
           Mean solar irradiance (mW/cm²/µm)

       Returns
       -------
       Rrs : array-like
           Remote sensing reflectance (sr⁻¹)
       """
       return nLw / F0
