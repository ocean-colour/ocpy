============================
Inherent Optical Properties
============================

.. module:: ocpy.iop
   :synopsis: Inherent optical property calculations

The ``iop`` module provides functions for calculating and modeling inherent optical
properties (IOPs) including absorption by CDOM, detritus, and minerals.

CDOM Absorption
---------------

.. module:: ocpy.iop.cdom
   :synopsis: CDOM absorption modeling

Colored Dissolved Organic Matter (CDOM) absorption is typically modeled using
exponential or power-law functions.

.. autofunction:: ocpy.iop.cdom.a_exp

.. autofunction:: ocpy.iop.cdom.a_pow

.. autofunction:: ocpy.iop.cdom.fit_exp_norm

.. autofunction:: ocpy.iop.cdom.fit_exp_tot

.. autofunction:: ocpy.iop.cdom.fit_pow

CDOM Models
^^^^^^^^^^^

**Exponential Model**

The standard CDOM absorption model:

.. math::

   a_{CDOM}(\\lambda) = a_{CDOM}(\\lambda_0) \\cdot \\exp(-S_{CDOM} \\cdot (\\lambda - \\lambda_0))

where:

* a_CDOM(λ₀) is the absorption at reference wavelength (typically 440 nm)
* S_CDOM is the spectral slope (typically 0.014-0.020 nm⁻¹)
* λ₀ is the reference wavelength

**Power-Law Model**

An alternative parameterization:

.. math::

   a_{CDOM}(\\lambda) = a_{CDOM}(\\lambda_0) \\cdot \\left(\\frac{\\lambda}{\\lambda_0}\\right)^S

where S is typically around -5 to -7.

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from ocpy.iop import cdom

   wavelengths = np.arange(350, 701, 5)

   # Generate CDOM spectrum with exponential model
   a_cdom = cdom.a_exp(wavelengths, S_CDOM=0.017, wave0=440)

   # Fit measured CDOM data
   measured_wave = np.array([350, 400, 440, 500, 550])
   measured_a = np.array([0.5, 0.25, 0.15, 0.08, 0.05])

   # Fit with fixed S_CDOM
   a440_fit = cdom.fit_exp_norm(measured_wave, measured_a)

   # Fit with free parameters
   a440, S_CDOM = cdom.fit_exp_tot(measured_wave, measured_a)
   print(f"a_CDOM(440) = {a440:.3f} m⁻¹, S = {S_CDOM:.4f} nm⁻¹")

ZLee IOP Methods
----------------

.. module:: ocpy.iop.zlee
   :synopsis: Zheng Lee IOP methods

The ZLee module implements IOP retrieval methods from the Zheng Lee suite of algorithms.

.. autofunction:: ocpy.iop.zlee.Y_from_Rrs

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from ocpy.iop.zlee import Y_from_Rrs

   wavelengths = np.array([412, 443, 490, 510, 555, 670])
   Rrs = np.array([0.003, 0.004, 0.005, 0.006, 0.007, 0.001])

   Y = Y_from_Rrs(wavelengths, Rrs)
   print(f"Y parameter: {Y:.3f}")

Particle Cross-Sections
-----------------------

.. module:: ocpy.iop.cross
   :synopsis: Particle cross-sections from Stramski et al. 2001

Functions for optical cross-sections of oceanic particles based on Stramski et al. (2001).

Detritus
^^^^^^^^

.. autofunction:: ocpy.iop.cross.detritus_abs

.. autofunction:: ocpy.iop.cross.detritus_scatt

.. autofunction:: ocpy.iop.cross.detritus_backscatt

Minerals
^^^^^^^^

.. autofunction:: ocpy.iop.cross.minerals_abs

.. autofunction:: ocpy.iop.cross.mineral_scatt

.. autofunction:: ocpy.iop.cross.mineral_backscatt

Bubbles
^^^^^^^

.. autofunction:: ocpy.iop.cross.bubbles_abs

.. autofunction:: ocpy.iop.cross.bubbles_scatt

.. autofunction:: ocpy.iop.cross.bubbles_backscatt

Cross-Section Background
^^^^^^^^^^^^^^^^^^^^^^^^

Optical cross-sections relate the concentration of particles to their optical effect:

.. math::

   a_{particles} = \\sum_i N_i \\cdot \\sigma_{a,i}

where:

* N_i is the number concentration of particle type i (particles/m³)
* σ_a,i is the absorption cross-section (m²/particle)

Similarly for scattering and backscattering.

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from ocpy.iop import cross

   wavelengths = np.arange(400, 701, 10)

   # Get cross-sections
   sigma_abs_det = cross.detritus_abs(wavelengths)
   sigma_scat_det = cross.detritus_scatt(wavelengths)

   # Calculate absorption from particle concentration
   N_detritus = 1e9  # particles/m³
   a_detritus = N_detritus * sigma_abs_det

IOP Relationships
-----------------

Common relationships between IOPs:

**Total Absorption**

.. math::

   a(\\lambda) = a_w(\\lambda) + a_{ph}(\\lambda) + a_{CDOM}(\\lambda) + a_{NAP}(\\lambda)

where:

* a_w = pure water absorption
* a_ph = phytoplankton absorption
* a_CDOM = CDOM absorption
* a_NAP = non-algal particle absorption

**Total Backscattering**

.. math::

   b_b(\\lambda) = b_{bw}(\\lambda) + b_{bp}(\\lambda)

where:

* b_bw = pure water backscattering
* b_bp = particulate backscattering

**Particulate Backscattering Ratio**

.. math::

   \\tilde{b}_{bp} = \\frac{b_{bp}}{b_p}

typically ranges from 0.5% to 3% depending on particle composition.

Spectral Slopes
---------------

IOP spectral slopes provide information about particle and dissolved matter composition:

**CDOM Slope (S_CDOM)**

* S_CDOM ≈ 0.014-0.016 nm⁻¹: Humic-dominated, terrestrial origin
* S_CDOM ≈ 0.018-0.022 nm⁻¹: Autochthonous, marine origin

**Particulate Backscattering Slope (η)**

The spectral slope of b_bp:

.. math::

   b_{bp}(\\lambda) = b_{bp}(\\lambda_0) \\cdot \\left(\\frac{\\lambda_0}{\\lambda}\\right)^\\eta

* η ≈ 0: Large particles (phytoplankton blooms)
* η ≈ 1-2: Small particles (bacteria, detritus)

References
----------

* Bricaud, A., Morel, A., and Prieur, L. (1981). Absorption by dissolved organic
  matter of the sea (yellow substance) in the UV and visible domains. Limnology
  and Oceanography, 26(1), 43-53.

* Stramski, D., Boss, E., Bogucki, D., and Voss, K.J. (2004). The role of seawater
  constituents in light backscattering in the ocean. Progress in Oceanography,
  61(1), 27-56.

* Lee, Z., Carder, K.L., and Arnone, R.A. (2002). Deriving inherent optical
  properties from water color: a multiband quasi-analytical algorithm for
  optically deep waters. Applied Optics, 41(27), 5755-5772.
