==============
IOP Inversions
==============

This guide covers deriving Inherent Optical Properties (IOPs) from remote sensing
reflectance using the LS2 and ZLee algorithms in ocpy.

Introduction
------------

IOP inversion algorithms solve the "inverse problem" of ocean color remote sensing:
given the measured Rrs spectrum, what are the underlying absorption and scattering
properties?

The main challenges are:

* The relationship between Rrs and IOPs is non-linear
* Multiple IOP combinations can produce similar Rrs
* Environmental factors (sun angle, wind) affect the signal

ocpy implements the LS2 model, which uses look-up tables from radiative transfer
simulations to address these challenges.

The LS2 Algorithm
-----------------

Algorithm Overview
^^^^^^^^^^^^^^^^^^

The LS2 (Loisel-Stramski version 2) algorithm:

1. Normalizes Rrs for sun angle effects
2. Matches the spectrum to pre-computed look-up tables
3. Retrieves absorption (a) and backscattering (bb)
4. Optionally corrects for Raman scattering

Setting Up LS2
^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from oceancolor.ls2 import ls2_main, io as ls2_io
   from oceancolor.ls2.kd_nn import Kd_NN_MODIS
   from oceancolor.water import absorption

   # Load the look-up tables (do this once)
   LUT = ls2_io.load_LUT()

   # Standard wavelengths (MODIS-like)
   wavelengths = np.array([412, 443, 490, 510, 555, 670])

   # Get pure water properties
   a_w = absorption.a_water(wavelengths, data='IOCCG')
   b_w = np.array([0.0058, 0.0045, 0.0031, 0.0026, 0.0019, 0.0008])

Running the Inversion
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Example Rrs spectrum (typical Case 1 water)
   Rrs = np.array([0.0025, 0.0032, 0.0045, 0.0052, 0.0058, 0.0008])
   sza = 30.0  # Solar zenith angle in degrees

   # First, estimate Kd using the neural network
   Kd = Kd_NN_MODIS(Rrs, sza, wavelengths)

   # Run LS2 inversion
   results = ls2_main.LS2_main(
       sza=sza,
       lambda_=wavelengths,
       Rrs=Rrs,
       Kd=Kd,
       aw=a_w,
       bw=b_w,
       bp=np.zeros_like(wavelengths),  # Set to 0 if unknown
       LS2_LUT=LUT,
       Flag_Raman=1  # 1 = include Raman correction
   )

   # Access results
   print("LS2 Results:")
   print(f"  Total absorption (a): {results['a']}")
   print(f"  Non-water absorption (anw): {results['anw']}")
   print(f"  Total backscattering (bb): {results['bb']}")
   print(f"  Particulate backscattering (bbp): {results['bbp']}")
   print(f"  Raman correction (kappa): {results['kappa']}")

Understanding the Outputs
^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: LS2 Output Variables
   :header-rows: 1
   :widths: 15 25 60

   * - Variable
     - Units
     - Description
   * - a
     - m⁻¹
     - Total absorption (water + constituents)
   * - anw
     - m⁻¹
     - Non-water absorption (a - a_w)
   * - bb
     - m⁻¹
     - Total backscattering (water + particles)
   * - bbp
     - m⁻¹
     - Particulate backscattering (bb - bb_w)
   * - kappa
     - dimensionless
     - Raman scattering correction factor

Kd Estimation
-------------

The diffuse attenuation coefficient (Kd) is required input for LS2. ocpy provides
a neural network for estimating Kd from Rrs.

.. code-block:: python

   from oceancolor.ls2.kd_nn import Kd_NN_MODIS, load_weights

   # The neural network works with MODIS-like wavelengths
   wavelengths = np.array([412, 443, 488, 531, 551, 667])
   Rrs = np.array([0.003, 0.004, 0.005, 0.006, 0.007, 0.001])
   sza = 30.0

   # Estimate Kd
   Kd = Kd_NN_MODIS(Rrs, sza, wavelengths)

   print(f"Estimated Kd: {Kd}")

   # The network automatically selects between clear and turbid models
   # based on Rrs characteristics

Processing Multiple Spectra
---------------------------

For satellite imagery with many pixels:

.. code-block:: python

   import numpy as np
   from oceancolor.ls2 import ls2_main, io as ls2_io
   from oceancolor.ls2.kd_nn import Kd_NN_MODIS
   from oceancolor.water import absorption

   # Load LUTs once
   LUT = ls2_io.load_LUT()

   def process_image(Rrs_cube, sza_array, wavelengths):
       """Process a satellite image through LS2.

       Parameters
       ----------
       Rrs_cube : ndarray
           3D array (rows, cols, wavelengths)
       sza_array : ndarray
           2D array of solar zenith angles
       wavelengths : ndarray
           Wavelength array

       Returns
       -------
       dict
           Dictionary of IOP arrays
       """
       nrows, ncols, nwaves = Rrs_cube.shape

       # Initialize output arrays
       a_out = np.full((nrows, ncols, nwaves), np.nan)
       bb_out = np.full((nrows, ncols, nwaves), np.nan)
       anw_out = np.full((nrows, ncols, nwaves), np.nan)
       bbp_out = np.full((nrows, ncols, nwaves), np.nan)

       # Get water properties
       a_w = absorption.a_water(wavelengths, data='IOCCG')
       b_w = np.array([0.0058, 0.0045, 0.0031, 0.0026, 0.0019, 0.0008])

       # Process each pixel
       for i in range(nrows):
           for j in range(ncols):
               Rrs = Rrs_cube[i, j, :]
               sza = sza_array[i, j]

               # Skip invalid pixels
               if np.any(np.isnan(Rrs)) or np.any(Rrs < 0):
                   continue

               try:
                   # Estimate Kd
                   Kd = Kd_NN_MODIS(Rrs, sza, wavelengths)

                   # Run LS2
                   results = ls2_main.LS2_main(
                       sza=sza,
                       lambda_=wavelengths,
                       Rrs=Rrs,
                       Kd=Kd,
                       aw=a_w,
                       bw=b_w,
                       bp=np.zeros_like(wavelengths),
                       LS2_LUT=LUT,
                       Flag_Raman=1
                   )

                   # Store results
                   a_out[i, j, :] = results['a']
                   bb_out[i, j, :] = results['bb']
                   anw_out[i, j, :] = results['anw']
                   bbp_out[i, j, :] = results['bbp']

               except Exception:
                   continue

       return {
           'a': a_out,
           'bb': bb_out,
           'anw': anw_out,
           'bbp': bbp_out
       }

ZLee Methods
------------

The ZLee suite provides alternative IOP retrieval methods:

.. code-block:: python

   from oceancolor.iop.zlee import Y_from_Rrs

   wavelengths = np.array([412, 443, 490, 510, 555, 670])
   Rrs = np.array([0.003, 0.004, 0.005, 0.006, 0.007, 0.001])

   # Calculate Y parameter (related to bb/(a+bb))
   Y = Y_from_Rrs(wavelengths, Rrs)
   print(f"Y parameter: {Y:.4f}")

Partitioning IOPs
-----------------

Once you have total IOPs, you may want to partition them into components:

.. code-block:: python

   from oceancolor.iop import cdom
   from oceancolor.ph import pigments
   import numpy as np

   def partition_absorption(wavelengths, a_nw, method='simple'):
       """Partition non-water absorption into components.

       Simple method assumes:
       a_nw = a_ph + a_dg
       where a_dg = CDOM + detrital (both have similar spectral shape)
       """
       # Estimate a_dg using long wavelength slope
       # (phytoplankton absorption is near zero above 650 nm)
       idx_red = wavelengths > 600

       if method == 'simple':
           # Fit exponential to red wavelengths
           S_dg = 0.015  # Typical detrital+CDOM slope

           # Use 443 nm as reference
           idx_443 = np.argmin(np.abs(wavelengths - 443))

           # Estimate a_dg(443) from red wavelengths
           # Extrapolate exponential back to 443
           a_red = np.mean(a_nw[idx_red])
           wv_red = np.mean(wavelengths[idx_red])
           a_dg_443 = a_red * np.exp(S_dg * (wv_red - 443))

           # Generate full a_dg spectrum
           a_dg = cdom.a_exp(wavelengths, S_CDOM=S_dg, wave0=443)
           a_dg = a_dg * (a_dg_443 / a_dg[idx_443])

           # Phytoplankton is the remainder
           a_ph = a_nw - a_dg
           a_ph[a_ph < 0] = 0  # Ensure non-negative

           return {'a_ph': a_ph, 'a_dg': a_dg}

Validation
----------

Always validate IOP retrievals against in-situ data when possible:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt

   def validate_iops(retrieved, measured, wavelength_idx=1):
       """Compare retrieved vs measured IOPs.

       Parameters
       ----------
       retrieved : array
           Retrieved IOP values
       measured : array
           In-situ measured IOP values
       wavelength_idx : int
           Wavelength index to plot (default 1 = 443 nm)
       """
       # Remove invalid data
       valid = ~np.isnan(retrieved) & ~np.isnan(measured)
       x = measured[valid]
       y = retrieved[valid]

       # Statistics
       log_x = np.log10(x)
       log_y = np.log10(y)
       bias = np.mean(log_y - log_x)
       rmse = np.sqrt(np.mean((log_y - log_x)**2))
       r = np.corrcoef(log_x, log_y)[0, 1]

       # Plot
       fig, ax = plt.subplots(figsize=(8, 8))

       ax.scatter(x, y, alpha=0.5, s=20)
       ax.plot([0.001, 10], [0.001, 10], 'k--', label='1:1')

       ax.set_xscale('log')
       ax.set_yscale('log')
       ax.set_xlabel('Measured (m⁻¹)')
       ax.set_ylabel('Retrieved (m⁻¹)')

       # Add statistics text
       text = f'Bias: {bias:.3f}\nRMSE: {rmse:.3f}\nR: {r:.3f}'
       ax.text(0.05, 0.95, text, transform=ax.transAxes,
               verticalalignment='top', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

       ax.legend()
       ax.grid(True, alpha=0.3)

       return {'bias': bias, 'rmse': rmse, 'r': r}

Best Practices
--------------

1. **Check input quality**:

   * Ensure Rrs is positive (negative values indicate atmospheric correction problems)
   * Check for cloud and land flags
   * Verify sun angle is within algorithm limits (typically < 70°)

2. **Use appropriate wavelengths**:

   * LS2 is optimized for specific bands
   * Interpolate hyperspectral data to standard wavelengths

3. **Consider algorithm limitations**:

   * Accuracy decreases in optically complex waters
   * Very low or very high IOPs may be out of LUT range
   * Shallow waters with bottom reflectance require special handling

4. **Apply quality flags**:

   .. code-block:: python

      def flag_iop_quality(a, bb, Rrs):
          """Generate quality flags for IOP retrievals."""
          flags = np.zeros_like(a[:, 0], dtype=int)

          # Flag negative IOPs
          flags[np.any(a < 0, axis=1)] |= 1

          # Flag extreme values
          flags[np.any(a > 5, axis=1)] |= 2
          flags[np.any(bb > 0.1, axis=1)] |= 4

          # Flag based on Rrs quality
          flags[np.any(Rrs < 0, axis=1)] |= 8

          return flags

References
----------

* Loisel, H., et al. (2018). An inverse model for estimating the optical absorption
  and backscattering coefficients of seawater from remote-sensing reflectance.
  JGR Oceans, 123, 2141-2171.

* Lee, Z., et al. (2002). Deriving inherent optical properties from water color:
  a multiband quasi-analytical algorithm for optically deep waters.
  Applied Optics, 41(27), 5755-5772.
