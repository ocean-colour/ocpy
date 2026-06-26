==========================
Pure Water Optical Properties
==========================

This guide covers the calculation of pure seawater optical properties using ocpy.

Introduction
------------

Pure water forms the optical baseline for ocean color remote sensing. Understanding
water's absorption and scattering properties is essential for:

* Correcting remote sensing data
* Separating water constituents from total optical properties
* Modeling radiative transfer in the ocean

Water Absorption
----------------

Pure water absorbs light across the visible spectrum, with very low absorption in
the blue (400-500 nm) and increasing strongly into the red (>600 nm).

Loading Reference Data
^^^^^^^^^^^^^^^^^^^^^^

ocpy provides two standard reference datasets:

.. code-block:: python

   import numpy as np
   from ocpy.water import absorption

   # Define wavelengths
   wavelengths = np.arange(400, 701, 5)

   # GSFC (NASA) reference data
   a_w_gsfc = absorption.a_water(wavelengths, data='GSFC')

   # IOCCG 2018 updated values (recommended)
   a_w_ioccg = absorption.a_water(wavelengths, data='IOCCG')

   # Compare the datasets
   import matplotlib.pyplot as plt

   plt.figure(figsize=(10, 6))
   plt.semilogy(wavelengths, a_w_gsfc, 'b-', label='GSFC', linewidth=2)
   plt.semilogy(wavelengths, a_w_ioccg, 'r--', label='IOCCG 2018', linewidth=2)
   plt.xlabel('Wavelength (nm)')
   plt.ylabel('Absorption coefficient (m⁻¹)')
   plt.legend()
   plt.title('Pure Water Absorption')
   plt.grid(True, alpha=0.3)

Key Wavelengths
^^^^^^^^^^^^^^^

Water absorption at key ocean color wavelengths:

.. list-table:: Pure Water Absorption
   :header-rows: 1
   :widths: 20 30 50

   * - Wavelength
     - a_w (m⁻¹)
     - Notes
   * - 412 nm
     - 0.0047
     - Minimal absorption
   * - 443 nm
     - 0.0071
     - Chlorophyll blue peak
   * - 490 nm
     - 0.0148
     - Band ratio algorithms
   * - 555 nm
     - 0.0565
     - Green reference
   * - 670 nm
     - 0.439
     - Chlorophyll red peak

Water Scattering
----------------

Seawater scattering is due to molecular (Rayleigh) scattering and is affected by
temperature and salinity.

Zhang et al. (2009) Model
^^^^^^^^^^^^^^^^^^^^^^^^^

The standard model for pure seawater scattering:

.. code-block:: python

   from ocpy.water.scattering import betasw_ZHH2009

   wavelengths = np.arange(400, 701, 10)

   # Standard conditions: T=20°C, S=35 PSU, 90° angle
   b_sw, bb_sw = betasw_ZHH2009(
       lambda_=wavelengths,
       Tc=20,      # Temperature in Celsius
       theta=90,   # Scattering angle in degrees
       S=35,       # Salinity in PSU
       delta=0.039 # Depolarization ratio
   )

   print(f"Scattering at 550 nm: {b_sw[wavelengths==550]:.4f} m⁻¹")
   print(f"Backscattering at 550 nm: {bb_sw[wavelengths==550]:.4f} m⁻¹")

Temperature and Salinity Effects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Water scattering varies with environmental conditions:

.. code-block:: python

   from ocpy.water.scattering import betasw_ZHH2009
   import numpy as np
   import matplotlib.pyplot as plt

   wavelengths = np.arange(400, 701, 10)

   # Temperature effect
   temps = [5, 15, 25]
   plt.figure(figsize=(10, 5))

   for T in temps:
       b_sw, _ = betasw_ZHH2009(wavelengths, Tc=T, theta=90, S=35)
       plt.plot(wavelengths, b_sw * 1000, label=f'T = {T}°C')

   plt.xlabel('Wavelength (nm)')
   plt.ylabel('Scattering coefficient (×10⁻³ m⁻¹)')
   plt.legend()
   plt.title('Temperature Effect on Water Scattering')
   plt.grid(True, alpha=0.3)

   # Salinity effect
   salinities = [0, 20, 35]
   plt.figure(figsize=(10, 5))

   for S in salinities:
       b_sw, _ = betasw_ZHH2009(wavelengths, Tc=20, theta=90, S=S)
       plt.plot(wavelengths, b_sw * 1000, label=f'S = {S} PSU')

   plt.xlabel('Wavelength (nm)')
   plt.ylabel('Scattering coefficient (×10⁻³ m⁻¹)')
   plt.legend()
   plt.title('Salinity Effect on Water Scattering')

Refractive Index
^^^^^^^^^^^^^^^^

The refractive index of seawater is needed for scattering calculations:

.. code-block:: python

   from ocpy.water.scattering import RInw

   wavelengths = np.array([400, 500, 600, 700])
   n_water = RInw(wavelengths, Tc=20, S=35)

   for wl, n in zip(wavelengths, n_water):
       print(f"n({wl} nm) = {n:.5f}")

   # Typical output:
   # n(400 nm) = 1.35029
   # n(500 nm) = 1.34372
   # n(600 nm) = 1.34088
   # n(700 nm) = 1.33903

Combining Absorption and Scattering
-----------------------------------

For ocean color modeling, you often need both properties together:

.. code-block:: python

   import numpy as np
   from ocpy.water import absorption, scattering

   def get_water_iops(wavelengths, T=20, S=35):
       """Get pure water inherent optical properties.

       Parameters
       ----------
       wavelengths : array-like
           Wavelengths in nm
       T : float
           Temperature in Celsius
       S : float
           Salinity in PSU

       Returns
       -------
       dict
           Dictionary with a_w, b_w, bb_w
       """
       a_w = absorption.a_water(wavelengths, data='IOCCG')
       b_w, bb_w = scattering.betasw_ZHH2009(wavelengths, Tc=T, theta=90, S=S)

       return {
           'wavelength': wavelengths,
           'a_w': a_w,
           'b_w': b_w,
           'bb_w': bb_w,
           'c_w': a_w + b_w  # Beam attenuation
       }

   # Example usage
   wave = np.arange(400, 701, 5)
   water_iops = get_water_iops(wave, T=22, S=36)

   # Plot all water IOPs
   fig, axes = plt.subplots(2, 2, figsize=(12, 10))

   axes[0, 0].semilogy(wave, water_iops['a_w'])
   axes[0, 0].set_ylabel('a_w (m⁻¹)')
   axes[0, 0].set_title('Water Absorption')

   axes[0, 1].plot(wave, water_iops['b_w'] * 1000)
   axes[0, 1].set_ylabel('b_w (×10⁻³ m⁻¹)')
   axes[0, 1].set_title('Water Scattering')

   axes[1, 0].plot(wave, water_iops['bb_w'] * 1000)
   axes[1, 0].set_ylabel('bb_w (×10⁻³ m⁻¹)')
   axes[1, 0].set_title('Water Backscattering')

   axes[1, 1].semilogy(wave, water_iops['c_w'])
   axes[1, 1].set_ylabel('c_w (m⁻¹)')
   axes[1, 1].set_title('Water Beam Attenuation')

   for ax in axes.flat:
       ax.set_xlabel('Wavelength (nm)')
       ax.grid(True, alpha=0.3)

   plt.tight_layout()

Applications
------------

Removing Water Contribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To isolate non-water absorption from total absorption:

.. code-block:: python

   # Total measured absorption
   a_total = np.array([...])  # Your measurements

   # Get water absorption
   a_w = absorption.a_water(wavelengths, data='IOCCG')

   # Non-water absorption
   a_nw = a_total - a_w

   # Typical components:
   # a_nw = a_ph + a_CDOM + a_NAP
   # where:
   #   a_ph = phytoplankton absorption
   #   a_CDOM = CDOM absorption
   #   a_NAP = non-algal particle absorption

Radiative Transfer Input
^^^^^^^^^^^^^^^^^^^^^^^^

Water IOPs are essential inputs for radiative transfer models:

.. code-block:: python

   def prepare_hydrolight_input(wavelengths, T, S, constituents):
       """Prepare IOPs for Hydrolight simulation.

       Parameters
       ----------
       wavelengths : array
           Wavelengths in nm
       T : float
           Temperature
       S : float
           Salinity
       constituents : dict
           Dictionary with a_ph, a_CDOM, a_NAP, bb_p

       Returns
       -------
       dict
           Complete IOP set for Hydrolight
       """
       # Pure water
       water = get_water_iops(wavelengths, T, S)

       return {
           'wavelength': wavelengths,
           'a_total': water['a_w'] + constituents['a_ph'] +
                      constituents['a_CDOM'] + constituents['a_NAP'],
           'bb_total': water['bb_w'] + constituents['bb_p'],
           'a_w': water['a_w'],
           'bb_w': water['bb_w']
       }

Best Practices
--------------

1. **Use IOCCG 2018 data** for most applications (most up-to-date)
2. **Account for T and S** when high accuracy is needed
3. **Match sensor wavelengths** exactly for satellite applications
4. **Document your water reference** for reproducibility

References
----------

* Pope, R.M. and Fry, E.S. (1997). Absorption spectrum (380-700 nm) of pure water.
  Applied Optics, 36(33), 8710-8723.

* Zhang, X., Hu, L., and He, M.-X. (2009). Scattering by pure seawater: Effect of
  salinity. Optics Express, 17(7), 5698-5710.

* IOCCG (2018). Earth Observations in Support of Global Water Quality Monitoring.
  IOCCG Report Series, No. 17.
