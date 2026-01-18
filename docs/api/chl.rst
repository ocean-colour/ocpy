======================
Chlorophyll Algorithms
======================

.. module:: oceancolor.chl
   :synopsis: Chlorophyll-a estimation algorithms

The ``chl`` module provides standard algorithms for estimating chlorophyll-a concentration
from remote sensing reflectance (Rrs).

Band Ratio Algorithms
---------------------

.. module:: oceancolor.chl.band_ratios
   :synopsis: OC2, OC4 band ratio algorithms

Band ratio algorithms are empirical relationships between spectral band ratios and
chlorophyll concentration, derived from global in-situ datasets.

.. autofunction:: oceancolor.chl.band_ratios.oc4

.. autofunction:: oceancolor.chl.band_ratios.oc2

Algorithm Description
^^^^^^^^^^^^^^^^^^^^^

**OC4 Algorithm**

The OC4 algorithm uses the maximum band ratio among three blue wavelengths:

.. math::

   R = \\log_{10}\\left(\\max\\left(\\frac{R_{rs}(443)}{R_{rs}(555)}, \\frac{R_{rs}(490)}{R_{rs}(555)}, \\frac{R_{rs}(510)}{R_{rs}(555)}\\right)\\right)

.. math::

   \\log_{10}(Chl) = a_0 + a_1 R + a_2 R^2 + a_3 R^3 + a_4 R^4

where the coefficients (a₀, a₁, a₂, a₃, a₄) are derived from NASA's ocean color
algorithm database.

**OC2 Algorithm**

The OC2 algorithm uses a single band ratio:

.. math::

   R = \\log_{10}\\left(\\frac{R_{rs}(490)}{R_{rs}(555)}\\right)

.. math::

   \\log_{10}(Chl) = a_0 + a_1 R + a_2 R^2 + a_3 R^3 + a_4 R^4

OC2 is simpler but less accurate than OC4, particularly at low chlorophyll concentrations.

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from oceancolor.chl.band_ratios import oc2, oc4

   # Define wavelengths and Rrs
   wave = np.array([443, 490, 510, 555, 670])
   Rrs = np.array([0.0045, 0.0052, 0.0055, 0.0060, 0.0012])

   # Calculate chlorophyll using both algorithms
   chl_oc4 = oc4(wave, Rrs)
   chl_oc2 = oc2(wave, Rrs)

   print(f"OC4 Chlorophyll: {chl_oc4:.3f} mg/m³")
   print(f"OC2 Chlorophyll: {chl_oc2:.3f} mg/m³")

Working with Arrays
^^^^^^^^^^^^^^^^^^^

The algorithms can process multiple spectra:

.. code-block:: python

   import numpy as np
   from oceancolor.chl.band_ratios import oc4

   # Multiple spectra (N x 5 array)
   wave = np.array([443, 490, 510, 555, 670])
   Rrs_array = np.array([
       [0.0045, 0.0052, 0.0055, 0.0060, 0.0012],  # Spectrum 1
       [0.0030, 0.0040, 0.0045, 0.0050, 0.0010],  # Spectrum 2
       [0.0060, 0.0065, 0.0068, 0.0070, 0.0015],  # Spectrum 3
   ])

   # Process all spectra
   for i, Rrs in enumerate(Rrs_array):
       chl = oc4(wave, Rrs)
       print(f"Spectrum {i+1}: Chl = {chl:.3f} mg/m³")

Algorithm Limitations
^^^^^^^^^^^^^^^^^^^^^

Band ratio algorithms have known limitations:

* **Low chlorophyll**: Less accurate below ~0.1 mg/m³
* **High chlorophyll**: Saturation effects above ~30 mg/m³
* **Optically complex waters**: Interference from CDOM and sediments
* **Atmospheric correction errors**: Sensitive to blue band errors

For challenging conditions, consider using:

* Semi-analytical algorithms (LS2, QAA)
* Red/NIR algorithms for turbid waters
* Algorithm blending approaches

Sensor-Specific Considerations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The algorithms expect specific wavelengths. For different sensors:

.. list-table:: Sensor Wavelength Mapping
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Algorithm Band
     - SeaWiFS
     - MODIS
     - VIIRS
     - PACE
   * - Blue 1
     - 443
     - 443
     - 443
     - 443
   * - Blue 2
     - 490
     - 488
     - 486
     - 490
   * - Blue 3
     - 510
     - 531
     - 551
     - 510
   * - Green
     - 555
     - 551
     - 551
     - 555

Interpolate Rrs to standard wavelengths if needed:

.. code-block:: python

   from scipy.interpolate import interp1d

   # Original sensor wavelengths
   sensor_wave = np.array([443, 488, 531, 551, 667])
   sensor_Rrs = np.array([0.0045, 0.0052, 0.0055, 0.0060, 0.0012])

   # Interpolate to standard wavelengths
   f = interp1d(sensor_wave, sensor_Rrs, kind='linear', fill_value='extrapolate')
   standard_wave = np.array([443, 490, 510, 555, 670])
   standard_Rrs = f(standard_wave)

   chl = oc4(standard_wave, standard_Rrs)

Quality Flags
^^^^^^^^^^^^^

Consider flagging unreliable retrievals:

.. code-block:: python

   def oc4_with_flags(wave, Rrs):
       """OC4 with quality flags."""
       from oceancolor.chl.band_ratios import oc4

       chl = oc4(wave, Rrs)

       # Quality flags
       flag = 0
       if chl < 0.01:
           flag |= 1  # Very low chlorophyll
       if chl > 100:
           flag |= 2  # Very high chlorophyll
       if Rrs[wave == 670].max() > Rrs[wave == 555].max():
           flag |= 4  # Red > green (sediments likely)
       if any(Rrs < 0):
           flag |= 8  # Negative Rrs (atmospheric correction issue)

       return chl, flag

References
----------

* O'Reilly, J.E., Maritorena, S., Mitchell, B.G., Siegel, D.A., Carder, K.L.,
  Garver, S.A., Kahru, M., and McClain, C. (1998). Ocean color chlorophyll
  algorithms for SeaWiFS. Journal of Geophysical Research, 103(C11), 24937-24953.

* O'Reilly, J.E. and Werdell, P.J. (2019). Chlorophyll algorithms for ocean color
  sensors - OC4, OC5 & OC6. Remote Sensing of Environment, 229, 32-47.

* NASA Ocean Biology Processing Group (2022). Chlorophyll-a Algorithm Theoretical
  Basis Document. https://oceancolor.gsfc.nasa.gov/atbd/chlor_a/
