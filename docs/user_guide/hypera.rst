=========================
Hyper-a Data Processing
=========================

This guide covers processing data from the Hyper-a integrating cavity absorption meter.

Introduction
------------

The Hyper-a (Sequoia Scientific) is a bench-top integrating cavity absorption meter
designed for measuring particulate absorption spectra with minimal scattering artifacts.

Key features:

* Hyperspectral coverage (380-710 nm, 2 nm resolution)
* Integrating cavity design minimizes scattering errors
* Suitable for both filtered and unfiltered samples
* Requires careful calibration and temperature/salinity corrections

Instrument Principles
---------------------

The integrating cavity measurement relies on multiple reflections within a highly
reflective sphere to achieve an effective path length much longer than the physical
cavity size.

.. code-block:: text

   Light source → Cavity (sample) → Multiple reflections → Detector

   Effective path length: L_eff = L / (1 - ρ)

   where:
   - L = physical path length
   - ρ = sphere wall reflectivity

Loading Data
------------

.. code-block:: python

   from ocpy.hyper_a import io as hypera_io

   # Load calibration file
   calibration = hypera_io.load_calibration('path/to/calibration.mat')

   print(f"Calibration wavelengths: {calibration.wavelengths[:10]}...")
   print(f"Sphere reflectivity: {calibration.rho:.4f}")

   # Load binary data file
   data = hypera_io.read_bin('path/to/sample.bin')

   print(f"Number of spectra: {data.n_spectra}")
   print(f"Integration time: {data.integration_time} ms")

   # Alternative: load from MATLAB format
   mat_data = hypera_io.load_mat_data('path/to/data.mat')

   # Universal import function
   hypera_data = hypera_io.import_hypera_data('path/to/any_format')

Basic Processing
----------------

.. code-block:: python

   from ocpy.hyper_a import io as hypera_io
   from ocpy.hyper_a import process

   # Load calibration and data
   cal = hypera_io.load_calibration('calibration.mat')
   data = hypera_io.read_bin('sample.bin')

   # Process with environmental parameters
   result = process.process(
       cal=cal,
       data=data,
       T=20.0,          # Water temperature (°C)
       S=35.0,          # Salinity (PSU)
       chl_correct=True # Apply chlorophyll fluorescence correction
   )

   # Access results
   wavelengths = result.wavelengths
   a_p = result.a_p  # Particulate absorption (m⁻¹)

   # Plot spectrum
   import matplotlib.pyplot as plt

   plt.figure(figsize=(10, 6))
   plt.plot(wavelengths, a_p)
   plt.xlabel('Wavelength (nm)')
   plt.ylabel('a_p (m⁻¹)')
   plt.title('Particulate Absorption Spectrum')
   plt.grid(True, alpha=0.3)

Processing Steps
----------------

The full processing workflow includes several corrections:

Linearity Correction
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from ocpy.hyper_a.lib import linearity_correct_pixels

   # Apply detector linearity correction
   corrected_data = linearity_correct_pixels(raw_data)

Dark Correction
^^^^^^^^^^^^^^^

.. code-block:: python

   from ocpy.hyper_a.lib import dark_correct_spectrum

   # Remove dark current
   corrected_spectrum = dark_correct_spectrum(data)

Pure Water Absorption
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from ocpy.hyper_a.lib import get_ioccg_aw

   # Get pure water absorption for subtraction
   wavelengths = np.arange(380, 711, 2)
   a_w = get_ioccg_aw(wavelengths, T=20, S=35)

   print(f"a_w at 440 nm: {a_w[wavelengths == 440]:.4f} m⁻¹")

Transmission and Absorption
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from ocpy.hyper_a.lib import compute_transmission, compute_absorption

   # Calculate transmission
   transmission = compute_transmission(
       sample_signal=sample_data,
       reference_signal=reference_data,
       dark_signal=dark_data
   )

   # Calculate absorption from transmission
   absorption = compute_absorption(
       transmission=transmission,
       path_length=cal.path_length,
       rho=cal.rho
   )

Chlorophyll Fluorescence Correction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Chlorophyll fluorescence near 685 nm can artificially lower apparent absorption:

.. code-block:: python

   from ocpy.hyper_a.lib import compute_chl_fluorescence_correction

   # Calculate fluorescence correction
   correction = compute_chl_fluorescence_correction(
       wavelengths=wavelengths,
       a_measured=a_measured,
       excitation_wavelength=440
   )

   # Apply correction
   a_corrected = a_measured - correction

Calibration
-----------

ND Spot Calibration
^^^^^^^^^^^^^^^^^^^

Sphere reflectivity is determined using neutral density filters:

.. code-block:: python

   from ocpy.hyper_a import process

   # Calculate reflectivity from ND spot measurement
   rho = process.rho_from_nd_spot(
       nd_data=nd_measurement_data,
       nd_transmission=0.1  # Known ND filter transmission
   )

   print(f"Calculated sphere reflectivity: {rho:.4f}")

Temperature Dependence
^^^^^^^^^^^^^^^^^^^^^^

Pure water absorption changes with temperature:

.. code-block:: python

   from ocpy.hyper_a.lib import get_ioccg_aw
   import numpy as np
   import matplotlib.pyplot as plt

   wavelengths = np.arange(380, 711, 2)

   temps = [5, 15, 25]
   plt.figure(figsize=(10, 6))

   for T in temps:
       a_w = get_ioccg_aw(wavelengths, T=T, S=35)
       plt.plot(wavelengths, a_w, label=f'T = {T}°C')

   plt.xlabel('Wavelength (nm)')
   plt.ylabel('a_w (m⁻¹)')
   plt.title('Pure Water Absorption vs Temperature')
   plt.legend()
   plt.grid(True, alpha=0.3)

Variable T/S Processing
-----------------------

For depth profiles with varying temperature and salinity:

.. code-block:: python

   from ocpy.hyper_a import process
   import numpy as np

   # CTD data
   depths = np.array([0, 10, 25, 50, 100, 200])
   temperatures = np.array([24, 22, 18, 15, 10, 5])
   salinities = np.array([35.0, 35.2, 35.5, 35.8, 36.0, 36.2])

   # Process with variable T/S
   results = process.process_with_variable_ts(
       cal=cal,
       data=data,
       depths=depths,
       T=temperatures,
       S=salinities
   )

   # Each depth gets appropriate T/S correction

Quality Control
---------------

.. code-block:: python

   import numpy as np

   def qc_hypera_spectrum(wavelengths, a_p, a_unc=None):
       """Quality control for Hyper-a spectra.

       Returns
       -------
       flags : dict
           Dictionary of quality flags
       """
       flags = {}

       # Check for negative values
       if np.any(a_p < -0.001):
           flags['negative_absorption'] = True
           # Small negatives might be noise; large negatives indicate problems

       # Check magnitude (typical range 0-5 m⁻¹)
       if np.any(a_p > 5):
           flags['high_absorption'] = True

       # Check for spectral anomalies
       # Fluorescence: should not have peak at 685 nm after correction
       idx_fl = (wavelengths > 670) & (wavelengths < 710)
       idx_ref = (wavelengths > 620) & (wavelengths < 660)

       if np.nanmean(a_p[idx_fl]) > 1.2 * np.nanmean(a_p[idx_ref]):
           flags['fluorescence_residual'] = True

       # Check spectral shape
       # ap should generally decrease with wavelength
       idx_blue = wavelengths < 450
       idx_red = wavelengths > 600

       if np.nanmean(a_p[idx_red]) > np.nanmean(a_p[idx_blue]):
           flags['unusual_spectral_shape'] = True

       # Check uncertainty if available
       if a_unc is not None:
           cv = np.nanmean(a_unc / np.abs(a_p) * 100)
           if cv > 50:
               flags['high_uncertainty'] = True

       return flags

   # Example usage
   flags = qc_hypera_spectrum(wavelengths, a_p)
   if flags:
       print(f"Quality issues: {list(flags.keys())}")
   else:
       print("Spectrum passed QC")

Batch Processing
----------------

.. code-block:: python

   import os
   import glob
   import numpy as np
   import pandas as pd
   from ocpy.hyper_a import io as hypera_io
   from ocpy.hyper_a import process

   def batch_process_hypera(data_dir, cal_file, output_file,
                            T=20.0, S=35.0):
       """Process multiple Hyper-a files.

       Parameters
       ----------
       data_dir : str
           Directory containing .bin files
       cal_file : str
           Path to calibration file
       output_file : str
           Output file path (CSV or NetCDF)
       T, S : float
           Temperature and salinity
       """
       # Load calibration
       cal = hypera_io.load_calibration(cal_file)

       # Find all data files
       data_files = sorted(glob.glob(os.path.join(data_dir, '*.bin')))
       print(f"Found {len(data_files)} files")

       results_list = []

       for f in data_files:
           try:
               # Load and process
               data = hypera_io.read_bin(f)
               result = process.process(
                   cal=cal, data=data, T=T, S=S
               )

               # Store results
               row = {
                   'filename': os.path.basename(f),
                   'wavelengths': result.wavelengths,
                   'a_p': result.a_p
               }
               results_list.append(row)

               print(f"Processed: {os.path.basename(f)}")

           except Exception as e:
               print(f"Error processing {f}: {e}")

       # Save results
       # ... (format-specific saving code)

       return results_list

Integration with Other Modules
------------------------------

Combining Hyper-a data with other ocpy functions:

.. code-block:: python

   from ocpy.hyper_a import process
   from ocpy.iop import cdom
   from ocpy.ph import pigments
   import numpy as np

   # Get Hyper-a particulate absorption
   result = process.process(cal=cal, data=data, T=20, S=35)
   wavelengths = result.wavelengths
   a_p = result.a_p

   # Partition into phytoplankton and detrital
   # Estimate detrital from exponential fit at long wavelengths
   idx_red = wavelengths > 600
   a_d_model = cdom.a_exp(wavelengths, S_CDOM=0.011, wave0=440)

   # Scale to match red wavelengths
   scale = np.nanmean(a_p[idx_red]) / np.nanmean(a_d_model[idx_red])
   a_d = a_d_model * scale

   # Phytoplankton absorption
   a_ph = a_p - a_d
   a_ph = np.maximum(a_ph, 0)

   # Estimate chlorophyll from a_ph
   # Using Bricaud relationship
   idx_443 = np.argmin(np.abs(wavelengths - 443))
   a_ph_443 = a_ph[idx_443]
   chl_est = (a_ph_443 / 0.04)**(1/0.7)

   print(f"Estimated chlorophyll: {chl_est:.2f} mg/m³")

References
----------

* Röttgers, R. and Doerffer, R. (2007). Measurements of optical absorption by
  chromophoric dissolved organic matter using a point-source integrating-cavity
  absorption meter. L&O Methods, 5, 126-135.

* Röttgers, R., McKee, D., and Woźniak, S.B. (2013). Evaluation of scatter
  corrections for ac-9 absorption measurements in coastal waters.
  Methods in Oceanography, 7, 21-39.

* Sequoia Scientific: https://www.sequoiasci.com/
