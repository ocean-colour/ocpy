==========
Quickstart
==========

This guide provides a quick introduction to the main features of ocpy.

Basic Concepts
--------------

Ocean color remote sensing relies on measuring the spectral distribution of light
leaving the ocean surface (remote sensing reflectance, Rrs) to infer water properties.

Key optical quantities in ocpy:

* **Rrs**: Remote sensing reflectance (sr\ :sup:`-1`)
* **a**: Absorption coefficient (m\ :sup:`-1`)
* **b**: Scattering coefficient (m\ :sup:`-1`)
* **bb**: Backscattering coefficient (m\ :sup:`-1`)
* **Kd**: Diffuse attenuation coefficient (m\ :sup:`-1`)

Wavelengths are always in nanometers (nm).

Pure Water Properties
---------------------

Get absorption and scattering coefficients for pure seawater:

.. code-block:: python

   import numpy as np
   from ocpy.water import absorption, scattering

   # Define wavelengths
   wavelengths = np.arange(400, 701, 10)  # 400-700 nm in 10 nm steps

   # Pure water absorption
   a_w = absorption.a_water(wavelengths, data='IOCCG')

   # Pure water scattering (Zhang et al. 2009 model)
   # Parameters: wavelength, temperature (C), angle (degrees), salinity (PSU)
   beta_w = scattering.betasw_ZHH2009(wavelengths, Tc=20, theta=90, S=35)

   # Total scattering and backscattering
   b_w, bb_w = scattering.betasw_ZHH2009(wavelengths, Tc=20, theta=90, S=35,
                                          delta=0.039)

Chlorophyll Estimation
----------------------

Estimate chlorophyll-a concentration from Rrs using band-ratio algorithms:

.. code-block:: python

   import numpy as np
   from ocpy.chl import band_ratios

   # Example Rrs at standard wavelengths
   wave = np.array([443, 490, 510, 555, 670])
   Rrs = np.array([0.0045, 0.0052, 0.0055, 0.0060, 0.0012])

   # OC4 algorithm (uses max band ratio)
   chl_oc4 = band_ratios.oc4(wave, Rrs)
   print(f"Chlorophyll-a (OC4): {chl_oc4:.2f} mg/m³")

   # OC2 algorithm (490/555 ratio only)
   chl_oc2 = band_ratios.oc2(wave, Rrs)
   print(f"Chlorophyll-a (OC2): {chl_oc2:.2f} mg/m³")

IOP Inversions with LS2
-----------------------

Derive inherent optical properties from Rrs using the LS2 model:

.. code-block:: python

   import numpy as np
   from ocpy.ls2 import ls2_main, io as ls2_io
   from ocpy.water import absorption

   # Load the LS2 look-up tables
   LUT = ls2_io.load_LUT()

   # Define inputs
   wavelengths = np.array([412, 443, 490, 510, 555, 670])
   sza = 30.0  # Solar zenith angle in degrees

   # Example Rrs spectrum
   Rrs = np.array([0.003, 0.004, 0.005, 0.006, 0.007, 0.001])

   # Get pure water properties
   a_w = absorption.a_water(wavelengths)
   b_w = np.array([0.0058, 0.0045, 0.0031, 0.0026, 0.0019, 0.0008])  # Approximate

   # Estimate Kd (diffuse attenuation)
   Kd = np.array([0.05, 0.04, 0.035, 0.03, 0.04, 0.5])  # Example values

   # Run LS2 inversion
   # bp is particulate scattering (set to 0 if unknown)
   results = ls2_main.LS2_main(
       sza=sza,
       lambda_=wavelengths,
       Rrs=Rrs,
       Kd=Kd,
       aw=a_w,
       bw=b_w,
       bp=np.zeros_like(wavelengths),
       LS2_LUT=LUT,
       Flag_Raman=1  # Include Raman correction
   )

   # Results include: a, anw, bb, bbp, kappa
   a_total = results['a']  # Total absorption
   bb_total = results['bb']  # Total backscattering
   print(f"Total absorption at 443 nm: {a_total[1]:.4f} m⁻¹")

Phytoplankton Absorption
------------------------

Work with phytoplankton absorption spectra:

.. code-block:: python

   import numpy as np
   from ocpy.ph import absorption as ph_abs
   from ocpy.ph import pigments

   # Load Bricaud 1998 reference data
   bricaud_data = ph_abs.load_bricaud1998()

   # Get chlorophyll-specific absorption spectrum
   wave = np.arange(400, 701, 5)
   a_chl_star = pigments.a_chl(wave, ctype='a', source='bricaud')

   # Scale by chlorophyll concentration
   chl = 1.0  # mg/m³
   a_ph = a_chl_star * chl

   print(f"a_ph at 443 nm: {a_ph[wave == 440][0] if 440 in wave else 'N/A':.4f} m⁻¹")

CDOM Absorption
---------------

Model CDOM (colored dissolved organic matter) absorption:

.. code-block:: python

   import numpy as np
   from ocpy.iop import cdom

   wavelengths = np.arange(350, 701, 5)

   # Exponential CDOM model
   # a_CDOM(λ) = a_CDOM(440) * exp(-S * (λ - 440))
   a_cdom_exp = cdom.a_exp(wavelengths, S_CDOM=0.017, wave0=440)

   # Power-law CDOM model
   a_cdom_pow = cdom.a_pow(wavelengths, S=-5.9, wave0=440)

   # Fit CDOM to measured data
   measured_a = np.array([...])  # Your measured CDOM absorption
   # a440, S_CDOM = cdom.fit_exp_tot(wavelengths, measured_a)

Working with Satellite Data
---------------------------

PACE OCI Data
^^^^^^^^^^^^^

Load and work with PACE satellite data:

.. code-block:: python

   from ocpy.pace import io as pace_io
   from ocpy.satellites import pace

   # Generate PACE wavelength array
   wave = pace.wave(wv_min=400, wv_max=700, step=5)

   # Generate PACE noise/error vector
   noise = pace.gen_noise_vector(wave, include_sampling=True)

   # Load PACE L2 data (if you have a file)
   # Rrs, Rrs_unc, FLH = pace_io.load_oci_l2('path/to/pace_file.nc')

MODIS/SeaWiFS Matchups
^^^^^^^^^^^^^^^^^^^^^^

Work with satellite matchup data:

.. code-block:: python

   from ocpy.satellites import modis, seawifs

   # Load MODIS matchup dataset
   modis_data = modis.load_matchups()

   # Calculate error statistics
   errors = modis.calc_errors(rel_in_situ_error=0.05)

Tara Oceans Data
----------------

Load and analyze Tara Oceans expedition data:

.. code-block:: python

   from ocpy.tara import io as tara_io
   from ocpy.tara import spectra, measures

   # Load the Tara database
   tara_db = tara_io.load_db(dataset='pg')  # Patrick Gray database

   # Extract absorption spectra
   wave, a_p = spectra.spectra_from_table(tara_db, flavor='ap')

   # Calculate derived quantities
   tara_db = measures.add_derived(tara_db, quantities=['chl', 'poc'])

Hydrolight Simulations
----------------------

Access Loisel+2023 Hydrolight radiative transfer simulations:

.. code-block:: python

   import os
   # First, ensure OS_COLOR environment variable is set
   # os.environ['OS_COLOR'] = '/path/to/loisel2023/data'

   from ocpy.hydrolight import loisel23

   # Load dataset
   # X: 1=no inelastic, 2=Raman, 4=Raman+fluorescence
   # Y: solar zenith angle (0, 30, or 60 degrees)
   ds = loisel23.load_ds(X=2, Y=30)

   # Access variables
   Rrs = ds['Rrs']
   wavelength = ds['wavelength']

   # Calculate chlorophyll from the simulation inputs
   chl = loisel23.calc_Chl(ds)

Hyper-a Processing
------------------

Process Hyper-a integrating cavity absorption meter data:

.. code-block:: python

   from ocpy.hyper_a import io as hypera_io
   from ocpy.hyper_a import process

   # Load calibration and data
   cal = hypera_io.load_calibration('path/to/calibration.mat')
   data = hypera_io.read_bin('path/to/data.bin')

   # Process the data
   result = process.process(
       cal=cal,
       data=data,
       T=20.0,  # Temperature in Celsius
       S=35.0   # Salinity in PSU
   )

   # Access results
   wavelengths = result.wavelengths
   absorption = result.a_p  # Particulate absorption

Plotting
--------

Create publication-quality plots:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from ocpy.water import absorption
   from ocpy.utils import plotting

   wavelengths = np.arange(400, 701, 5)
   a_w = absorption.a_water(wavelengths, data='IOCCG')

   fig, ax = plt.subplots(figsize=(10, 6))
   ax.semilogy(wavelengths, a_w, 'b-', linewidth=2)
   ax.set_xlabel('Wavelength (nm)')
   ax.set_ylabel('Absorption (m⁻¹)')
   ax.set_title('Pure Water Absorption Spectrum')
   ax.set_xlim(400, 700)
   ax.grid(True, alpha=0.3)

   # Set consistent font sizes
   plotting.set_fontsize(ax, 12)

   plt.tight_layout()
   plt.show()

Next Steps
----------

* See the :doc:`User Guide <user_guide/water_optics>` for detailed tutorials
* Browse the :doc:`API Reference <api/water>` for complete function documentation
* Check out example notebooks in the ``examples/`` directory
