=======================
Water Optical Properties
=======================

.. module:: ocpy.water
   :synopsis: Pure seawater optical properties

The ``water`` module provides functions for calculating the optical properties of pure
seawater, including absorption and scattering coefficients.

Absorption
----------

.. module:: ocpy.water.absorption
   :synopsis: Pure water absorption coefficients

Functions for calculating pure water absorption coefficients using various reference datasets.

.. autofunction:: ocpy.water.absorption.a_water

.. autofunction:: ocpy.water.absorption.load_rsr_gsfc

.. autofunction:: ocpy.water.absorption.load_ioccg_2018

Reference Data
^^^^^^^^^^^^^^

The module includes two main reference datasets:

**GSFC Dataset**

The Goddard Space Flight Center (GSFC) reference standard for pure water absorption,
commonly used in NASA ocean color processing.

**IOCCG 2018 Dataset**

Updated pure water absorption values from the International Ocean Colour Coordinating
Group (IOCCG) 2018 compilation, which represents the current best estimates.

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from ocpy.water.absorption import a_water, load_ioccg_2018

   # Get absorption at specific wavelengths
   wavelengths = np.array([412, 443, 490, 510, 555, 670])
   a_w = a_water(wavelengths, data='IOCCG')
   print(f"Pure water absorption: {a_w}")

   # Load the full IOCCG dataset
   ioccg_data = load_ioccg_2018()
   print(f"Available wavelengths: {ioccg_data['wavelength'].min()}-{ioccg_data['wavelength'].max()} nm")

Scattering
----------

.. module:: ocpy.water.scattering
   :synopsis: Pure water scattering coefficients

Functions for calculating pure water scattering using the Zhang et al. (2009) model.

.. autofunction:: ocpy.water.scattering.betasw_ZHH2009

.. autofunction:: ocpy.water.scattering.RInw

.. autofunction:: ocpy.water.scattering.BetaT

.. autofunction:: ocpy.water.scattering.rhou_sw

.. autofunction:: ocpy.water.scattering.dlnasw_ds

.. autofunction:: ocpy.water.scattering.PMH

.. autofunction:: ocpy.water.scattering.bbw_from_l23

Zhang et al. (2009) Model
^^^^^^^^^^^^^^^^^^^^^^^^^

The Zhang-Hu-He 2009 model is the standard for calculating seawater scattering. It accounts
for:

* Temperature effects on water density and refractive index
* Salinity effects on ionic strength and molecular polarizability
* Wavelength dependence following Rayleigh scattering theory
* Depolarization ratio for anisotropic scattering

Key parameters:

* **λ (lambda)**: Wavelength in nanometers
* **Tc**: Temperature in degrees Celsius
* **S**: Salinity in PSU (Practical Salinity Units)
* **θ (theta)**: Scattering angle in degrees
* **δ (delta)**: Depolarization ratio (default 0.039)

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from ocpy.water.scattering import betasw_ZHH2009, RInw

   # Calculate scattering at 90 degrees (for backscattering)
   wavelengths = np.arange(400, 701, 10)
   beta_90 = betasw_ZHH2009(wavelengths, Tc=20, theta=90, S=35)

   # Get refractive index
   n_water = RInw(wavelengths, Tc=20, S=35)

   # Total scattering and backscattering
   b_sw, bb_sw = betasw_ZHH2009(wavelengths, Tc=20, theta=90, S=35, delta=0.039)

Physical Background
^^^^^^^^^^^^^^^^^^^

Pure water scattering arises from density fluctuations at the molecular level. In seawater,
scattering is enhanced by concentration fluctuations due to dissolved salts.

The volume scattering function β(θ) describes the angular distribution of scattered light:

.. math::

   \\beta(\\theta) = \\beta_{90} \\cdot \\frac{1 + \\frac{1-\\delta}{1+\\delta} \\cos^2(\\theta)}{2}

where:

* β₉₀ is the scattering at 90 degrees
* δ is the depolarization ratio
* θ is the scattering angle

The backscattering coefficient is obtained by integrating over the backward hemisphere:

.. math::

   b_b = 2\\pi \\int_{\\pi/2}^{\\pi} \\beta(\\theta) \\sin(\\theta) d\\theta

References
----------

* Pope, R.M. and Fry, E.S. (1997). Absorption spectrum (380-700 nm) of pure water.
  II. Integrating cavity measurements. Applied Optics, 36(33), 8710-8723.

* Zhang, X., Hu, L., and He, M.-X. (2009). Scattering by pure seawater: Effect of
  salinity. Optics Express, 17(7), 5698-5710.

* IOCCG (2018). Earth Observations in Support of Global Water Quality Monitoring.
  IOCCG Report Series, No. 17.
