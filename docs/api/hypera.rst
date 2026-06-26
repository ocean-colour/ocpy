=========================
Hyper-a Absorption Meter
=========================

.. module:: ocpy.hyper_a
   :synopsis: Hyper-a integrating cavity absorption meter

The ``hyper_a`` module provides tools for processing data from the Hyper-a integrating
cavity absorption meter manufactured by Sequoia Scientific.

Overview
--------

The Hyper-a is an integrating cavity absorption meter that measures particulate and
dissolved absorption across visible wavelengths (380-710 nm). It uses an integrating
sphere design that minimizes scattering errors.

Key features:

* Hyperspectral coverage (2 nm resolution)
* Minimal scattering sensitivity
* Temperature and salinity corrections
* Chlorophyll fluorescence correction

Data I/O
--------

.. module:: ocpy.hyper_a.io
   :synopsis: Hyper-a file reading and calibration

Data Classes
^^^^^^^^^^^^

.. autoclass:: ocpy.hyper_a.io.HyperaConfig
   :members:
   :undoc-members:

.. autoclass:: ocpy.hyper_a.io.HyperaCalibration
   :members:
   :undoc-members:

.. autoclass:: ocpy.hyper_a.io.HyperaData
   :members:
   :undoc-members:

I/O Functions
^^^^^^^^^^^^^

.. autofunction:: ocpy.hyper_a.io.read_bin

.. autofunction:: ocpy.hyper_a.io.load_calibration

.. autofunction:: ocpy.hyper_a.io.load_mat_data

.. autofunction:: ocpy.hyper_a.io.import_hypera_data

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

   from ocpy.hyper_a import io as hypera_io

   # Load calibration file
   calibration = hypera_io.load_calibration('path/to/calibration.mat')

   # Read binary data file
   data = hypera_io.read_bin('path/to/data.bin')

   # Import data from various sources
   hypera_data = hypera_io.import_hypera_data('path/to/data')

Core Processing
---------------

.. module:: ocpy.hyper_a.lib
   :synopsis: Hyper-a core processing functions

Optical Functions
^^^^^^^^^^^^^^^^^

.. autofunction:: ocpy.hyper_a.lib.get_ioccg_aw

.. autofunction:: ocpy.hyper_a.lib.ps

.. autofunction:: ocpy.hyper_a.lib.compute_transmission

.. autofunction:: ocpy.hyper_a.lib.compute_absorption

.. autofunction:: ocpy.hyper_a.lib.compute_rho

Data Corrections
^^^^^^^^^^^^^^^^

.. autofunction:: ocpy.hyper_a.lib.linearity_correct_pixels

.. autofunction:: ocpy.hyper_a.lib.dark_correct_spectrum

.. autofunction:: ocpy.hyper_a.lib.interpolate_pixels_to_cal_wls

.. autofunction:: ocpy.hyper_a.lib.get_median_of_filter_runs

.. autofunction:: ocpy.hyper_a.lib.compute_chl_fluorescence_correction

Processing Theory
^^^^^^^^^^^^^^^^^

The integrating cavity measurement is based on the relationship:

.. math::

   T = \\exp(-a \\cdot L_{eff})

where:

* T is the measured transmission
* a is the absorption coefficient
* L_eff is the effective path length in the integrating cavity

The absorption is computed from:

.. math::

   a = \\frac{1}{L_{eff}} \\cdot \\ln\\left(\\frac{I_0}{I}\\right)

Main Processing Workflow
------------------------

.. module:: ocpy.hyper_a.process
   :synopsis: Hyper-a main processing workflow

Result Container
^^^^^^^^^^^^^^^^

.. autoclass:: ocpy.hyper_a.process.HyperaResult
   :members:
   :undoc-members:

Processing Functions
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: ocpy.hyper_a.process.process

.. autofunction:: ocpy.hyper_a.process.rho_from_nd_spot

.. autofunction:: ocpy.hyper_a.process.process_with_variable_ts

Example Processing
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from ocpy.hyper_a import io as hypera_io
   from ocpy.hyper_a import process

   # Load calibration and data
   cal = hypera_io.load_calibration('calibration.mat')
   data = hypera_io.read_bin('sample_data.bin')

   # Process with environmental corrections
   result = process.process(
       cal=cal,
       data=data,
       T=20.0,    # Temperature in Celsius
       S=35.0,    # Salinity in PSU
       chl_correct=True  # Apply chlorophyll fluorescence correction
   )

   # Access results
   print(f"Wavelengths: {result.wavelengths}")
   print(f"Particulate absorption shape: {result.a_p.shape}")

   # Plot spectrum
   import matplotlib.pyplot as plt
   plt.plot(result.wavelengths, result.a_p)
   plt.xlabel('Wavelength (nm)')
   plt.ylabel('a_p (m⁻¹)')
   plt.title('Particulate Absorption Spectrum')

Advanced Processing
^^^^^^^^^^^^^^^^^^^

For variable temperature and salinity during a cast:

.. code-block:: python

   from ocpy.hyper_a import process
   import numpy as np

   # CTD data along the cast
   depths = np.array([0, 5, 10, 20, 50, 100])  # meters
   temperatures = np.array([25, 24, 22, 18, 12, 8])  # Celsius
   salinities = np.array([35.0, 35.1, 35.2, 35.5, 35.8, 36.0])  # PSU

   # Process with variable T/S
   results = process.process_with_variable_ts(
       cal=cal,
       data=data,
       depths=depths,
       T=temperatures,
       S=salinities
   )

Calibration
-----------

The Hyper-a requires regular calibration to maintain accuracy:

**Sphere Reflectivity**

The integrating sphere reflectivity (ρ) is determined using neutral density filters:

.. code-block:: python

   from ocpy.hyper_a import process

   # Calculate reflectivity from ND spot calibration
   rho = process.rho_from_nd_spot(
       nd_data=nd_spot_data,
       nd_transmission=0.1  # Known ND filter transmission
   )

**Pure Water Reference**

The instrument is referenced against pure water absorption:

.. code-block:: python

   from ocpy.hyper_a.lib import get_ioccg_aw

   # Get IOCCG pure water absorption
   wavelengths = np.arange(380, 711, 2)
   a_w = get_ioccg_aw(wavelengths, T=20, S=0)

Corrections
-----------

**Linearity Correction**

Corrects for detector non-linearity at high signal levels:

.. code-block:: python

   from ocpy.hyper_a.lib import linearity_correct_pixels

   corrected_data = linearity_correct_pixels(raw_data)

**Dark Correction**

Removes dark current and stray light:

.. code-block:: python

   from ocpy.hyper_a.lib import dark_correct_spectrum

   corrected_spectrum = dark_correct_spectrum(data)

**Chlorophyll Fluorescence Correction**

Removes fluorescence emission near 685 nm:

.. code-block:: python

   from ocpy.hyper_a.lib import compute_chl_fluorescence_correction

   correction = compute_chl_fluorescence_correction(
       wavelengths=wavelengths,
       a_measured=a_measured,
       excitation_wavelength=440
   )
   a_corrected = a_measured - correction

Quality Control
---------------

Recommended quality control checks:

1. **Signal level**: Ensure adequate detector counts (>1000, <60000)
2. **Stability**: Check for drift during measurement
3. **Filter blank**: Verify filtrate shows expected near-zero absorption
4. **Spectral shape**: Check for anomalous features

.. code-block:: python

   def qc_hypera_spectrum(wavelengths, a_p, a_unc=None):
       """Basic quality control for Hyper-a spectra."""
       flags = {}

       # Check for negative values
       flags['negative'] = np.any(a_p < 0)

       # Check magnitude (typical range 0-5 m⁻¹)
       flags['high_absorption'] = np.any(a_p > 5)

       # Check for spectral anomalies near fluorescence
       fl_region = (wavelengths > 670) & (wavelengths < 710)
       if np.any(a_p[fl_region] > a_p[wavelengths == 650]):
           flags['fluorescence_artifact'] = True

       return flags

References
----------

* Röttgers, R. and Doerffer, R. (2007). Measurements of optical absorption by
  chromophoric dissolved organic matter using a point-source integrating-cavity
  absorption meter. Limnology and Oceanography: Methods, 5, 126-135.

* Röttgers, R., McKee, D., and Woźniak, S.B. (2013). Evaluation of scatter
  corrections for ac-9 absorption measurements in coastal waters. Methods in
  Oceanography, 7, 21-39.

* Sequoia Scientific: https://www.sequoiasci.com/
