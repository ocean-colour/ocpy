=========================
Phytoplankton Absorption
=========================

This guide covers working with phytoplankton absorption spectra and pigment analysis in ocpy.

Introduction
------------

Phytoplankton absorption is a key component of ocean optics, providing information about:

* Phytoplankton biomass (related to chlorophyll-a)
* Community composition (different pigments)
* Primary productivity estimation

ocpy provides tools for loading reference spectra, modeling absorption, and decomposing
measured spectra into pigment components.

Loading Reference Data
----------------------

Bricaud et al. (1998)
^^^^^^^^^^^^^^^^^^^^^

The standard model relating chlorophyll to phytoplankton absorption:

.. code-block:: python

   import numpy as np
   from ocpy.ph.absorption import load_bricaud1998

   # Load the Bricaud 1998 dataset
   bricaud = load_bricaud1998()

   # Access coefficients A(λ) and B(λ)
   wavelengths = bricaud['wavelength']
   A = bricaud['A']
   B = bricaud['B']

   print(f"Wavelength range: {wavelengths.min()}-{wavelengths.max()} nm")

   # Calculate absorption for a given chlorophyll concentration
   chl = 1.0  # mg/m³

   # Bricaud model: a*_ph(λ) = A(λ) * [Chl]^(-B(λ))
   a_ph_star = A * chl**(-B)  # Chlorophyll-specific absorption (m²/mg)
   a_ph = a_ph_star * chl     # Total phytoplankton absorption (m⁻¹)

   # Plot
   import matplotlib.pyplot as plt
   plt.figure(figsize=(10, 6))
   plt.plot(wavelengths, a_ph)
   plt.xlabel('Wavelength (nm)')
   plt.ylabel('a_ph (m⁻¹)')
   plt.title(f'Phytoplankton Absorption for Chl = {chl} mg/m³')

Other Datasets
^^^^^^^^^^^^^^

.. code-block:: python

   from ocpy.ph import load_data

   # Stramski 2001 - single cell measurements
   stramski = load_data.stramski2001()

   # Clementson 2019 - updated chlorophyll absorption
   clementson = load_data.clementson2019()

   # Moore 1995 - phytoplankton species
   moore = load_data.moore1995()

Chlorophyll-Specific Absorption
-------------------------------

The relationship between chlorophyll and absorption varies with phytoplankton community:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from ocpy.ph.absorption import load_bricaud1998

   bricaud = load_bricaud1998()
   wavelengths = bricaud['wavelength']
   A = bricaud['A']
   B = bricaud['B']

   # Calculate absorption for different chlorophyll levels
   chl_values = [0.1, 0.5, 1.0, 5.0, 10.0]

   fig, axes = plt.subplots(1, 2, figsize=(14, 5))

   # Plot a_ph
   ax = axes[0]
   for chl in chl_values:
       a_ph = A * chl**(1 - B)  # Simplification of A * chl^(-B) * chl
       ax.plot(wavelengths, a_ph, label=f'Chl = {chl} mg/m³')

   ax.set_xlabel('Wavelength (nm)')
   ax.set_ylabel('a_ph (m⁻¹)')
   ax.set_title('Phytoplankton Absorption')
   ax.legend()
   ax.set_xlim(400, 700)

   # Plot a*_ph (chlorophyll-specific)
   ax = axes[1]
   for chl in chl_values:
       a_ph_star = A * chl**(-B)
       ax.plot(wavelengths, a_ph_star, label=f'Chl = {chl} mg/m³')

   ax.set_xlabel('Wavelength (nm)')
   ax.set_ylabel('a*_ph (m²/mg)')
   ax.set_title('Chlorophyll-Specific Absorption')
   ax.legend()
   ax.set_xlim(400, 700)

   plt.tight_layout()

The Package Effect
^^^^^^^^^^^^^^^^^^

Larger phytoplankton cells show reduced chlorophyll-specific absorption due to
"packaging" of pigments within chloroplasts:

.. code-block:: python

   def package_effect_factor(cell_size, chl_intracellular):
       """Estimate the package effect factor.

       Q*_a = (3/2) * (1/ρ) * (1 + (ρ-1)*exp(-ρ) - (1-ρ)*(1-exp(-ρ))/ρ)

       where ρ = a_sol * d (optical depth through cell)
       """
       import numpy as np

       # Simplified model
       # a_sol = intracellular pigment absorption
       # d = cell diameter

       rho = chl_intracellular * cell_size * 0.01  # Optical depth (simplified)

       if rho < 0.01:
           return 1.0  # No packaging
       elif rho > 10:
           return 1.5 / rho  # Strong packaging
       else:
           Q_star = (3/2) * (1/rho) * (1 + (rho-1)*np.exp(-rho) -
                   (1-rho)*(1-np.exp(-rho))/rho)
           return Q_star

Pigment Analysis
----------------

Gaussian Decomposition
^^^^^^^^^^^^^^^^^^^^^^

Phytoplankton absorption can be decomposed into Gaussian pigment peaks:

.. code-block:: python

   import numpy as np
   from ocpy.ph import pigments

   wavelengths = np.arange(400, 701, 2)

   # Load chlorophyll absorption spectrum
   a_chl = pigments.a_chl(wavelengths, ctype='a', source='bricaud')

   # Generate individual pigment Gaussians
   # Index 0 = Chlorophyll-a (main peak)
   gauss_chla = pigments.gauss_pigment(wavelengths, idx=0)

   # Plot
   plt.figure(figsize=(10, 6))
   plt.plot(wavelengths, a_chl, 'k-', linewidth=2, label='Total')
   plt.plot(wavelengths, gauss_chla, 'g--', label='Chl-a Gaussian')
   plt.xlabel('Wavelength (nm)')
   plt.ylabel('Absorption')
   plt.legend()
   plt.title('Phytoplankton Absorption Decomposition')

Spectral Fitting
^^^^^^^^^^^^^^^^

Fit measured absorption spectra to extract pigment information:

.. code-block:: python

   from ocpy.ph import pigments
   import numpy as np

   # Measured phytoplankton absorption
   wavelengths = np.arange(400, 701, 5)
   a_ph_measured = np.array([...])  # Your measured spectrum

   # Fit with Gaussian components
   fit_result = pigments.fit_a_chl(
       wavelengths,
       a_ph_measured,
       fit_type='free',  # or 'constrained'
       add_pigments=None,  # or ['phycoerythrin', 'phycocyanin']
       sigma=None  # Measurement uncertainties
   )

   print(f"Fit result: {fit_result}")

Pigment Diagnostic Features
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Different pigments have characteristic absorption peaks:

.. list-table:: Diagnostic Pigment Wavelengths
   :header-rows: 1
   :widths: 25 20 55

   * - Pigment
     - Peak (nm)
     - Associated Phytoplankton
   * - Chlorophyll-a
     - 440, 675
     - All phytoplankton
   * - Chlorophyll-b
     - 470, 650
     - Green algae, prochlorophytes
   * - Chlorophyll-c
     - 460
     - Diatoms, dinoflagellates
   * - Fucoxanthin
     - 490
     - Diatoms
   * - Peridinin
     - 530
     - Dinoflagellates
   * - Phycoerythrin
     - 545, 565
     - Cyanobacteria, cryptophytes
   * - Phycocyanin
     - 620
     - Cyanobacteria

Phytoplankton Functional Types
------------------------------

Use absorption spectra to infer phytoplankton community composition:

.. code-block:: python

   import numpy as np

   def detect_pft(wavelengths, a_ph, a_ph_443=None):
       """Simple phytoplankton functional type detection.

       Based on absorption spectral shape and diagnostic features.
       """
       # Normalize spectrum
       if a_ph_443 is None:
           idx_443 = np.argmin(np.abs(wavelengths - 443))
           a_ph_443 = a_ph[idx_443]

       a_ph_norm = a_ph / a_ph_443

       # Find key indices
       idx_490 = np.argmin(np.abs(wavelengths - 490))
       idx_550 = np.argmin(np.abs(wavelengths - 550))
       idx_620 = np.argmin(np.abs(wavelengths - 620))
       idx_675 = np.argmin(np.abs(wavelengths - 675))

       pft_indicators = {}

       # Cyanobacteria: phycocyanin absorption near 620 nm
       # Creates a local maximum in normalized spectrum
       if a_ph_norm[idx_620] > 0.3:
           pft_indicators['cyanobacteria'] = a_ph_norm[idx_620]

       # Diatoms: high fucoxanthin, elevated 490 nm
       ratio_490_443 = a_ph_norm[idx_490]
       if ratio_490_443 > 0.8:
           pft_indicators['diatoms'] = ratio_490_443

       # Red tide (dinoflagellates): elevated 530-550 nm (peridinin)
       ratio_550_443 = a_ph_norm[idx_550]
       if ratio_550_443 > 0.4:
           pft_indicators['dinoflagellates'] = ratio_550_443

       # Picoplankton: narrow red peak at 675 nm
       red_peak_width = estimate_peak_width(wavelengths, a_ph, 675)
       if red_peak_width < 20:  # nm
           pft_indicators['picoplankton'] = 1.0 / red_peak_width

       return pft_indicators

   def estimate_peak_width(wavelengths, spectrum, peak_center):
       """Estimate FWHM of absorption peak."""
       idx_peak = np.argmin(np.abs(wavelengths - peak_center))
       peak_value = spectrum[idx_peak]
       half_max = peak_value / 2

       # Find half-max points
       above_half = spectrum > half_max
       transitions = np.diff(above_half.astype(int))

       # Simplified width estimate
       width_indices = np.where(above_half)[0]
       if len(width_indices) > 1:
           width = wavelengths[width_indices[-1]] - wavelengths[width_indices[0]]
       else:
           width = 50  # Default

       return width

From a_ph to Chlorophyll
------------------------

Estimate chlorophyll from phytoplankton absorption:

.. code-block:: python

   import numpy as np
   from scipy.optimize import minimize_scalar

   def chl_from_a_ph(wavelengths, a_ph, method='bricaud'):
       """Estimate chlorophyll from phytoplankton absorption.

       Inverts the Bricaud relationship.
       """
       from ocpy.ph.absorption import load_bricaud1998

       if method == 'bricaud':
           bricaud = load_bricaud1998()

           # Interpolate to measurement wavelengths
           from scipy.interpolate import interp1d
           f_A = interp1d(bricaud['wavelength'], bricaud['A'])
           f_B = interp1d(bricaud['wavelength'], bricaud['B'])

           A = f_A(wavelengths)
           B = f_B(wavelengths)

           # Minimize difference between modeled and measured
           def objective(log_chl):
               chl = 10**log_chl
               a_ph_model = A * chl**(1 - B)
               return np.sum((a_ph - a_ph_model)**2)

           result = minimize_scalar(objective, bounds=(-2, 3), method='bounded')
           chl = 10**result.x

           return chl

       elif method == 'simple':
           # Simple ratio at 443 nm
           idx_443 = np.argmin(np.abs(wavelengths - 443))
           a_ph_443 = a_ph[idx_443]

           # Typical relationship: Chl ≈ (a_ph(443) / 0.04)^(1/0.7)
           chl = (a_ph_443 / 0.04)**(1/0.7)

           return chl

Combining with IOP Inversions
-----------------------------

Integrate phytoplankton absorption with LS2 retrievals:

.. code-block:: python

   import numpy as np
   from ocpy.ls2 import ls2_main, io as ls2_io
   from ocpy.iop import cdom

   def retrieve_a_ph(wavelengths, Rrs, sza):
       """Retrieve phytoplankton absorption from Rrs.

       Uses LS2 for total absorption, then partitions.
       """
       from ocpy.water import absorption
       from ocpy.ls2.kd_nn import Kd_NN_MODIS

       # Load LUT
       LUT = ls2_io.load_LUT()

       # Pure water
       a_w = absorption.a_water(wavelengths, data='IOCCG')
       b_w = np.array([0.0058, 0.0045, 0.0031, 0.0026, 0.0019, 0.0008])

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

       # Non-water absorption
       a_nw = results['anw']

       # Partition: a_nw = a_ph + a_dg
       # Estimate a_dg from exponential fit at red wavelengths
       idx_red = wavelengths > 600
       a_dg_spectrum = cdom.a_exp(wavelengths, S_CDOM=0.015, wave0=443)

       # Scale a_dg to match red wavelengths
       scale = np.mean(a_nw[idx_red]) / np.mean(a_dg_spectrum[idx_red])
       a_dg = a_dg_spectrum * scale

       # Phytoplankton is remainder
       a_ph = a_nw - a_dg
       a_ph = np.maximum(a_ph, 0)  # Non-negative

       return a_ph, a_dg

References
----------

* Bricaud, A., et al. (1998). Variations of light absorption by suspended particles
  with chlorophyll a concentration in oceanic (case 1) waters. JGR, 103(C13), 31033-31044.

* Chase, A.P., et al. (2013). Decomposition of in situ particulate absorption spectra.
  Methods in Oceanography, 7, 110-124.

* Morel, A. and Bricaud, A. (1981). Theoretical results concerning light absorption
  in a discrete medium, and application to specific absorption of phytoplankton.
  Deep Sea Research Part A, 28(11), 1375-1393.
