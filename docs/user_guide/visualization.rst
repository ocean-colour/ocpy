======================
Ocean Color Visualization
======================

This guide covers tools for visualizing ocean color satellite data using ocpy.

Overview
--------

ocpy provides interactive visualization tools for ocean color granules from multiple sensors:

* **PACE OCI**: Hyperspectral Level 1C and Level 2 products
* **MODIS Aqua**: Multispectral Level 2 products
* Additional sensors through the standardized interface

Visualizations use Bokeh for interactive browser-based displays with pan, zoom, and hover capabilities.

Scene Visualization
-------------------

The ``ocpy_view`` command-line tool creates multi-panel displays of ocean color scenes,
showing multiple wavelength channels simultaneously.

Basic Usage
^^^^^^^^^^^

Display a PACE L2 granule with default wavelength channels:

.. code-block:: bash

   ocpy_view PACE_OCI.20240401.L2.OC.nc

This displays 9 wavelength channels by default: 350, 400, 452, 552, 600, 650, 700, 750, 801 nm.

Custom Wavelength Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Specify custom wavelength channels:

.. code-block:: bash

   # Display 4 channels across visible spectrum
   ocpy_view PACE_OCI.L2.nc --channels 400,450,550,650

   # Display red-edge and NIR channels
   ocpy_view PACE_OCI.L2.nc --channels 650,665,680,710,750

   # Single channel display
   ocpy_view PACE_OCI.L2.nc --channels 443

The tool automatically finds the closest available wavelength in the granule for each
requested channel.

Layout Configuration
^^^^^^^^^^^^^^^^^^^^

Control the grid layout:

.. code-block:: bash

   # 4 columns instead of default 3
   ocpy_view PACE_OCI.L2.nc --ncols 4

   # 2 columns for side-by-side comparison
   ocpy_view PACE_OCI.L2.nc --channels 443,555 --ncols 2

Saving Output
^^^^^^^^^^^^^

Save the interactive plot to an HTML file:

.. code-block:: bash

   ocpy_view PACE_OCI.L2.nc -o scene_visualization.html

   # Can be opened later in any web browser
   firefox scene_visualization.html

The HTML file is self-contained and can be shared or archived.

PACE L1C Data
^^^^^^^^^^^^^

View top-of-atmosphere data from Level 1C files:

.. code-block:: bash

   # Display TOA reflectance (rhot) or radiance (Lt)
   ocpy_view PACE_OCI.20240401.L1C.nc --channels 400,500,600,700,800

L1C files automatically use ``rhot`` (reflectance) if available, otherwise ``Lt`` (radiance).

MODIS Data
^^^^^^^^^^

View MODIS Aqua Level 2 products:

.. code-block:: bash

   # MODIS has fixed bands - requests will be matched to closest bands
   ocpy_view MODIS_A.20240401.L2.nc --channels 412,443,488,555,667

   # Request wavelengths will be matched to MODIS bands:
   # 410 → 412 nm, 445 → 443 nm, 490 → 488 nm, etc.

MODIS Available Bands: 412, 443, 469, 488, 531, 547, 555, 645, 667, 678 nm

Data Masking
^^^^^^^^^^^^

Invalid and flagged pixels are automatically displayed in gray:

* Pixels with non-zero L2 quality flags
* NaN or infinite values
* Atmospheric correction failures
* Land, cloud, or high-glint pixels

Interactive Features
^^^^^^^^^^^^^^^^^^^^

The browser display includes:

* **Pan**: Click and drag to move around the scene
* **Zoom**: Scroll wheel or box zoom tool
* **Hover**: See pixel coordinates and values
* **Color scales**: Each panel auto-scales to data range (1-99 percentile)
* **Export**: Save individual panels as PNG

Spectral Visualization
----------------------

The ``ocpy_plot_rrs`` tool displays individual pixel spectra with uncertainties.

View Pixel Spectrum
^^^^^^^^^^^^^^^^^^^

Extract and plot spectrum at a specific location:

.. code-block:: bash

   # By latitude/longitude
   ocpy_plot_rrs PACE_OCI.L2.nc --lat 34.5 --lon -120.3

   # By pixel coordinates
   ocpy_plot_rrs PACE_OCI.L2.nc --x 500 --y 800

   # Save to file
   ocpy_plot_rrs PACE_OCI.L2.nc --lat 34.5 --lon -120.3 -o spectrum.html

Features:

* Displays full hyperspectral Rrs curve
* Shows uncertainty bands (Rrs ± Rrs_unc)
* Toggle between linear and log y-axis scales
* Hover tooltips with wavelength and values
* Reports L2 quality flags

Programmatic Access
-------------------

Use the visualization functions in Python scripts:

Multi-Panel Scene Plot
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from ocpy.scripts import plot_oc_scene

   # Load and plot PACE L2 scene
   wavelengths = [400, 450, 500, 550, 600, 650, 700]

   data, lats, lons, flags, actual_wls = plot_oc_scene.load_pace_l2_scene(
       'PACE_OCI.L2.nc', wavelengths
   )

   plot_oc_scene.create_scene_plot(
       data, lats, lons, flags, actual_wls,
       filename='PACE_OCI.L2.nc',
       output_html='my_scene.html',
       ncols=3
   )

Automatic File Type Detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from ocpy.scripts import plot_oc_scene

   # Automatically detect file format
   file_type = plot_oc_scene.detect_file_type('unknown_granule.nc')

   if file_type == 'PACE_L2':
       print("PACE Level 2 OC file detected")
   elif file_type == 'PACE_L1C':
       print("PACE Level 1C file detected")
   elif file_type == 'MODIS_L2':
       print("MODIS Level 2 file detected")

Custom Processing
^^^^^^^^^^^^^^^^^

Load data for custom analysis:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from ocpy.scripts import plot_oc_scene

   # Load PACE L2 data
   wavelengths = [443, 555, 670]
   data, lats, lons, flags, wls = plot_oc_scene.load_pace_l2_scene(
       'PACE_OCI.L2.nc', wavelengths
   )

   # data shape: (n_wavelengths, nx, ny)
   # Apply custom masking
   valid = flags == 0  # Only perfectly valid pixels

   # Compute statistics
   for i, wl in enumerate(wls):
       channel_data = data[i, :, :]
       mean_rrs = np.nanmean(channel_data[valid])
       print(f"{wl:.1f} nm: mean Rrs = {mean_rrs:.6f} sr⁻¹")

   # Extract subset region
   lat_min, lat_max = 30.0, 35.0
   lon_min, lon_max = -125.0, -120.0

   mask = (lats >= lat_min) & (lats <= lat_max) & \
          (lons >= lon_min) & (lons <= lon_max)

   subset_data = data[:, mask]

   # Custom matplotlib plot
   fig, ax = plt.subplots()
   im = ax.imshow(data[0, :, :], cmap='viridis')
   plt.colorbar(im, label='Rrs (sr⁻¹)')
   ax.set_title(f'Rrs at {wls[0]:.1f} nm')
   plt.show()

Single Spectrum Plot
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from ocpy.scripts import plot_rrs

   # Load and plot spectrum at a location
   args_dict = {
       'granule': 'PACE_OCI.L2.nc',
       'lat': 34.5,
       'lon': -120.3,
       'x': None,
       'y': None,
       'output': 'spectrum.html'
   }

   import argparse
   args = argparse.Namespace(**args_dict)

   # This will display in browser and save HTML
   plot_rrs.main(args)

Advanced Visualization
----------------------

RGB Composites
^^^^^^^^^^^^^^

Create true-color or false-color composites:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from ocpy.scripts import plot_oc_scene

   # Load RGB channels
   rgb_wavelengths = [665, 555, 443]  # Red, Green, Blue
   data, lats, lons, flags, wls = plot_oc_scene.load_pace_l2_scene(
       'PACE_OCI.L2.nc', rgb_wavelengths
   )

   # Stack as RGB (need to transpose and normalize)
   rgb = np.stack([data[0], data[1], data[2]], axis=-1)

   # Normalize to 0-1 range
   rgb_norm = rgb - np.nanpercentile(rgb, 1, axis=(0,1))
   rgb_norm = rgb_norm / np.nanpercentile(rgb_norm, 99, axis=(0,1))
   rgb_norm = np.clip(rgb_norm, 0, 1)

   # Apply gamma correction for visibility
   rgb_gamma = rgb_norm ** 0.6

   # Mask invalid pixels as white
   valid = (flags == 0)[:, :, np.newaxis]
   rgb_masked = np.where(valid, rgb_gamma, 1.0)

   # Display
   fig, ax = plt.subplots(figsize=(12, 8))
   ax.imshow(rgb_masked)
   ax.set_title('True Color Composite')
   ax.axis('off')
   plt.tight_layout()
   plt.show()

Comparison Plots
^^^^^^^^^^^^^^^^

Compare multiple granules or processing levels:

.. code-block:: python

   from ocpy.scripts import plot_oc_scene
   import matplotlib.pyplot as plt
   import numpy as np

   # Load same region from L1C and L2
   wavelength = [555]

   data_l1c, lats1, lons1, flags1, wl1 = plot_oc_scene.load_pace_l1c_scene(
       'PACE_OCI.L1C.nc', wavelength
   )

   data_l2, lats2, lons2, flags2, wl2 = plot_oc_scene.load_pace_l2_scene(
       'PACE_OCI.L2.OC.nc', wavelength
   )

   # Create side-by-side comparison
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

   im1 = ax1.imshow(data_l1c[0], cmap='viridis')
   ax1.set_title('L1C: TOA Reflectance')
   plt.colorbar(im1, ax=ax1)

   im2 = ax2.imshow(data_l2[0], cmap='viridis')
   ax2.set_title('L2: Rrs (Atmospherically Corrected)')
   plt.colorbar(im2, ax=ax2)

   plt.tight_layout()
   plt.show()

Animation
^^^^^^^^^

Create animations from time series:

.. code-block:: python

   import matplotlib.pyplot as plt
   import matplotlib.animation as animation
   from glob import glob
   from ocpy.scripts import plot_oc_scene

   # Get list of granule files (sorted by time)
   files = sorted(glob('PACE_OCI.*.L2.nc'))

   # Load first file to set up plot
   wavelength = [555]
   data, lats, lons, flags, wl = plot_oc_scene.load_pace_l2_scene(
       files[0], wavelength
   )

   # Set up figure
   fig, ax = plt.subplots(figsize=(10, 8))
   im = ax.imshow(data[0], cmap='viridis', vmin=0, vmax=0.02)
   plt.colorbar(im, label='Rrs (sr⁻¹)')
   title = ax.set_title(f'Frame 0: {files[0]}')

   def update_frame(i):
       """Update animation frame"""
       data, _, _, _, _ = plot_oc_scene.load_pace_l2_scene(
           files[i], wavelength
       )
       im.set_array(data[0])
       title.set_text(f'Frame {i}: {files[i]}')
       return [im, title]

   # Create animation
   anim = animation.FuncAnimation(
       fig, update_frame, frames=len(files),
       interval=500, blit=True
   )

   # Save as movie
   # anim.save('chlorophyll_timeseries.mp4', writer='ffmpeg', fps=2)

   plt.show()

Performance Tips
----------------

Memory Management
^^^^^^^^^^^^^^^^^

For large granules:

.. code-block:: python

   # Load specific spatial subset
   from netCDF4 import Dataset
   import numpy as np

   with Dataset('PACE_OCI.L2.nc', 'r') as ds:
       gd = ds.groups['geophysical_data']

       # Load subset instead of full array
       i_start, i_end = 100, 500  # Along-track indices
       j_start, j_end = 200, 700  # Cross-track indices

       rrs_subset = gd.variables['Rrs'][i_start:i_end, j_start:j_end, :]

       # Process subset only
       print(f"Subset shape: {rrs_subset.shape}")

Parallel Processing
^^^^^^^^^^^^^^^^^^^

Process multiple granules in parallel:

.. code-block:: python

   from multiprocessing import Pool
   from ocpy.scripts import plot_oc_scene

   def process_granule(filename):
       """Process single granule"""
       wavelengths = [443, 555, 670]
       try:
           data, lats, lons, flags, wls = plot_oc_scene.load_pace_l2_scene(
               filename, wavelengths
           )
           # Do processing...
           return filename, True
       except Exception as e:
           return filename, False

   # Process in parallel
   files = ['file1.nc', 'file2.nc', 'file3.nc']
   with Pool(processes=4) as pool:
       results = pool.map(process_granule, files)

   for filename, success in results:
       print(f"{filename}: {'OK' if success else 'FAILED'}")

Troubleshooting
---------------

Common Issues
^^^^^^^^^^^^^

**"No valid wavelength channels found"**

The requested wavelengths may be outside the sensor's spectral range:

* PACE: 340-890 nm (hyperspectral)
* MODIS: Fixed bands at 412, 443, 469, 488, 531, 547, 555, 645, 667, 678 nm

Check the granule's wavelength coverage and adjust your channel selection.

**"All pixels are gray/masked"**

The region may have quality issues:

.. code-block:: python

   from netCDF4 import Dataset
   import numpy as np

   with Dataset('PACE_OCI.L2.nc', 'r') as ds:
       flags = ds.groups['geophysical_data'].variables['l2_flags'][:]

       # Check flag distribution
       unique, counts = np.unique(flags, return_counts=True)
       for flag_val, count in zip(unique, counts):
           pct = 100 * count / flags.size
           print(f"Flag {flag_val}: {pct:.1f}% of pixels")

       # Percentage of good pixels (flag==0)
       good_pct = 100 * np.sum(flags == 0) / flags.size
       print(f"Valid pixels: {good_pct:.1f}%")

**Memory errors on large files**

Use spatial subsetting or reduce the number of wavelengths:

.. code-block:: bash

   # Use fewer channels
   ocpy_view large_granule.nc --channels 443,555,670

References
----------

* Bokeh Documentation: https://docs.bokeh.org/
* PACE Data User Guide: https://oceancolor.gsfc.nasa.gov/data/pace/
* MODIS Ocean Color: https://oceancolor.gsfc.nasa.gov/data/aqua/

See Also
--------

* :doc:`satellites` - Working with satellite data formats
* :doc:`iop_inversions` - Deriving IOPs for visualization
* ``ocpy_view --help`` - Full command-line options
* ``ocpy_plot_rrs --help`` - Spectrum plotting options
