=============
Phytoplankton
=============

.. module:: ocpy.ph
   :synopsis: Phytoplankton optical properties

The ``ph`` module provides functions for phytoplankton absorption spectra, pigment
analysis, and community composition.

Absorption Spectra
------------------

.. module:: ocpy.ph.absorption
   :synopsis: Phytoplankton absorption coefficients

Functions for loading and calculating phytoplankton absorption spectra.

.. autofunction:: ocpy.ph.absorption.load_bricaud1998

Bricaud et al. (1998) Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Bricaud model relates chlorophyll-specific absorption to chlorophyll concentration:

.. math::

   a_{ph}^*(\\lambda) = A(\\lambda) \\cdot [Chl]^{-B(\\lambda)}

where:

* a*_ph(λ) is the chlorophyll-specific absorption (m²/mg Chl)
* A(λ) and B(λ) are wavelength-dependent coefficients
* [Chl] is the chlorophyll-a concentration (mg/m³)

The total phytoplankton absorption is:

.. math::

   a_{ph}(\\lambda) = a_{ph}^*(\\lambda) \\cdot [Chl]

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

   from ocpy.ph.absorption import load_bricaud1998
   import numpy as np

   # Load Bricaud 1998 data
   bricaud = load_bricaud1998()

   # Get wavelengths and coefficients
   wavelengths = bricaud['wavelength']
   A = bricaud['A']
   B = bricaud['B']

   # Calculate absorption for Chl = 1 mg/m³
   chl = 1.0
   a_ph_star = A * chl**(-B)
   a_ph = a_ph_star * chl

Data Loading
------------

.. module:: ocpy.ph.load_data
   :synopsis: Load phytoplankton reference data

Functions for loading various phytoplankton absorption datasets.

.. autofunction:: ocpy.ph.load_data.stramski2001

.. autofunction:: ocpy.ph.load_data.clementson2019

.. autofunction:: ocpy.ph.load_data.bricaud

.. autofunction:: ocpy.ph.load_data.moore1995

Available Datasets
^^^^^^^^^^^^^^^^^^

**Stramski et al. (2001)**

Optical properties of individual phytoplankton cells measured with flow cytometry.
Includes absorption, scattering, and backscattering for various species.

**Clementson (2019)**

Updated chlorophyll absorption spectra from in-situ measurements.

**Bricaud et al. (1998, 2004)**

Global relationships between chlorophyll and absorption from extensive field campaigns.

**Moore et al. (1995)**

Phytoplankton absorption spectra from various taxonomic groups.

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

   from ocpy.ph import load_data

   # Load Stramski data
   stramski = load_data.stramski2001()

   # Load Clementson chlorophyll absorption
   clementson = load_data.clementson2019()

   # Load Moore phytoplankton data
   moore = load_data.moore1995()

Pigment Analysis
----------------

.. module:: ocpy.ph.pigments
   :synopsis: Pigment absorption and fitting

Functions for pigment absorption spectra and spectral decomposition.

.. autofunction:: ocpy.ph.pigments.gauss_pigment

.. autofunction:: ocpy.ph.pigments.a_chl

.. autofunction:: ocpy.ph.pigments.fit_a_chl

Gaussian Pigment Model
^^^^^^^^^^^^^^^^^^^^^^

Pigment absorption can be modeled as a sum of Gaussian peaks (Chase et al. 2013):

.. math::

   a_{ph}(\\lambda) = \\sum_i A_i \\exp\\left(-\\frac{(\\lambda - \\lambda_i)^2}{2\\sigma_i^2}\\right)

where each Gaussian represents a pigment absorption band:

* Chlorophyll-a: peaks near 440 and 675 nm
* Chlorophyll-b: peaks near 470 and 650 nm
* Chlorophyll-c: peak near 460 nm
* Carotenoids: broad peak near 490 nm
* Phycoerythrin: peak near 565 nm
* Phycocyanin: peak near 620 nm

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from ocpy.ph import pigments

   wavelengths = np.arange(400, 701, 2)

   # Get chlorophyll absorption spectrum
   a_chl_star = pigments.a_chl(wavelengths, ctype='a', source='bricaud')

   # Generate Gaussian pigment profiles
   gauss_chl = pigments.gauss_pigment(wavelengths, idx=0)  # Chlorophyll-a

   # Fit measured phytoplankton absorption
   measured_wave = np.arange(400, 701, 5)
   measured_a_ph = np.array([...])  # Your measured spectrum

   fit_result = pigments.fit_a_chl(
       measured_wave,
       measured_a_ph,
       fit_type='free',
       sigma=None
   )

Spectral Decomposition
^^^^^^^^^^^^^^^^^^^^^^

Decompose phytoplankton absorption into pigment components:

.. code-block:: python

   import numpy as np
   from ocpy.ph import pigments

   # Measured phytoplankton absorption
   wavelengths = np.arange(400, 701, 2)
   a_ph_measured = np.array([...])

   # Fit with Gaussian components
   result = pigments.fit_a_chl(
       wavelengths,
       a_ph_measured,
       fit_type='free',
       add_pigments=['phycoerythrin', 'phycocyanin']  # Add accessory pigments
   )

   # Access fitted parameters
   print(f"Fitted parameters: {result}")

Phytoplankton Size Classes
--------------------------

Absorption spectra vary with phytoplankton size due to the package effect:

* **Picoplankton** (< 2 μm): High a*_ph, minimal packaging
* **Nanoplankton** (2-20 μm): Moderate packaging
* **Microplankton** (> 20 μm): Strong packaging, lower a*_ph

The package effect is described by:

.. math::

   a_{ph}^* = a_{sol}^* \\cdot Q_a^*

where:

* a*_sol is the absorption coefficient of pigment in solution
* Q*_a is the absorption efficiency factor (< 1 for packaging)

Functional Types
----------------

Different phytoplankton groups have characteristic absorption features:

.. list-table:: Phytoplankton Absorption Features
   :header-rows: 1
   :widths: 25 25 50

   * - Group
     - Key Wavelengths
     - Diagnostic Features
   * - Diatoms
     - 440, 490, 675
     - High Chl-c, fucoxanthin
   * - Coccolithophores
     - 440, 470, 675
     - High calcification backscatter
   * - Cyanobacteria
     - 440, 620, 675
     - Phycocyanin peak at 620 nm
   * - Prochlorococcus
     - 440, 480, 675
     - Divinyl chlorophyll
   * - Dinoflagellates
     - 440, 530, 675
     - Peridinin peak near 530 nm

References
----------

* Bricaud, A., Babin, M., Morel, A., and Claustre, H. (1995). Variability in the
  chlorophyll-specific absorption coefficients of natural phytoplankton: Analysis
  and parameterization. Journal of Geophysical Research, 100(C7), 13321-13332.

* Bricaud, A., Morel, A., Babin, M., Allali, K., and Claustre, H. (1998).
  Variations of light absorption by suspended particles with chlorophyll a
  concentration in oceanic (case 1) waters: Analysis and implications for
  bio-optical models. Journal of Geophysical Research, 103(C13), 31033-31044.

* Chase, A.P., Boss, E., Zaneveld, R., Bricaud, A., Claustre, H., Ras, J.,
  Dall'Olmo, G., and Westberry, T.K. (2013). Decomposition of in situ particulate
  absorption spectra. Methods in Oceanography, 7, 110-124.

* Moore, L.R., Goericke, R., and Chisholm, S.W. (1995). Comparative physiology of
  Synechococcus and Prochlorococcus: influence of light and temperature on growth,
  pigments, fluorescence and absorptive properties. Marine Ecology Progress
  Series, 116, 259-275.
