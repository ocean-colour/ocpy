===================
LS2 Inversion Model
===================

.. module:: ocpy.ls2
   :synopsis: LS2 model for IOP retrieval from Rrs

The ``ls2`` module implements the LS2 (Loisel-Stramski version 2) semi-analytical algorithm
for retrieving inherent optical properties (IOPs) from remote sensing reflectance (Rrs).

Overview
--------

The LS2 model (Loisel et al. 2018) derives:

* **a**: Total absorption coefficient (m⁻¹)
* **anw**: Non-water absorption coefficient (m⁻¹)
* **bb**: Total backscattering coefficient (m⁻¹)
* **bbp**: Particulate backscattering coefficient (m⁻¹)
* **κ (kappa)**: Raman scattering correction factor

The algorithm uses pre-computed look-up tables (LUTs) based on radiative transfer simulations
to relate Rrs to IOPs, accounting for:

* Solar zenith angle effects
* Raman scattering by water molecules
* Bidirectional reflectance effects

Main Algorithm
--------------

.. module:: ocpy.ls2.ls2_main
   :synopsis: Core LS2 inversion algorithm

.. autofunction:: ocpy.ls2.ls2_main.ls2_invert

.. autoclass:: ocpy.ls2.ls2_main.LS2Result
   :members: converged, counts

.. autofunction:: ocpy.ls2.ls2_main.LS2_main

.. autofunction:: ocpy.ls2.ls2_main.LS2_seek_pos

.. autofunction:: ocpy.ls2.ls2_main.LS2_calc_kappa

Algorithm Details
^^^^^^^^^^^^^^^^^

The LS2 algorithm operates in several steps:

1. **Normalization**: Rrs is normalized by solar zenith angle effects
2. **LUT Interpolation**: The normalized Rrs is matched to pre-computed LUT entries
3. **IOP Retrieval**: Absorption and backscattering are derived from the best-matching LUT entry
4. **Raman Correction**: An optional correction factor (κ) accounts for Raman scattered light

``ls2_invert`` is the entry point to use.  It runs over an ``(N, L)`` block of samples and
wavelengths at once, iterates the Raman correction to convergence, and reports why any cell
came back NaN through four per-cell flags rather than through a warning per cell.  ``LS2_main``
is a thin scalar wrapper kept for its historical 5-tuple contract; it defaults to a single
Raman pass, which is what the authors' MATLAB distribution does, and so remains a faithful
oracle for it.

Three behaviours differ deliberately from the earlier Python port, all bug fixes:

* All four ``a`` corners are recomputed in the Raman branch.  The port recomputed one and
  reused three stale corners; the authors' ``LS2_main.m`` recomputes all four.
* κ above the Raman table's 702 nm limit is NaN, not the clamped end-of-table value.
* Off-grid inputs return NaN rather than ``None``, and a value landing exactly on the last
  ``eta`` or ``muw`` node no longer raises ``UnboundLocalError``.

Input Requirements
^^^^^^^^^^^^^^^^^^

* **sza**: Solar zenith angle (degrees, 0-70)
* **lambda_**: Wavelength array (nm)
* **Rrs**: Remote sensing reflectance (sr⁻¹)
* **Kd**: Diffuse attenuation coefficient (m⁻¹) - can be estimated using the Kd_NN module
* **aw**: Pure water absorption (m⁻¹)
* **bw**: Pure water scattering (m⁻¹)
* **bp**: Particulate scattering (m⁻¹) - set to zeros if unknown
* **LS2_LUT**: Look-up tables loaded via ``load_LUT()``
* **Flag_Raman**: 0=no Raman, 1=with Raman correction

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from ocpy.ls2 import ls2_main, io as ls2_io
   from ocpy.water import absorption

   # Load LUTs
   LUT = ls2_io.load_LUT()

   # Define inputs
   wavelengths = np.array([412, 443, 490, 510, 555, 670])
   sza = 30.0
   Rrs = np.array([0.003, 0.004, 0.005, 0.006, 0.007, 0.001])
   Kd = np.array([0.05, 0.04, 0.035, 0.03, 0.04, 0.5])
   a_w = absorption.a_water(wavelengths)
   b_w = np.array([0.0058, 0.0045, 0.0031, 0.0026, 0.0019, 0.0008])

   # Run inversion over one spectrum (or an (N, L) block of them)
   res = ls2_main.ls2_invert(
       Rrs, Kd, a_w, b_w, np.zeros_like(wavelengths, dtype=float),
       sza, wavelengths, LUT, raman=True)

   print(f"Total absorption: {res.a}")
   print(f"Backscattering: {res.bb}")
   print(f"Flags: {res.counts()}")

I/O Functions
-------------

.. module:: ocpy.ls2.io
   :synopsis: LS2 look-up table loading

.. autofunction:: ocpy.ls2.io.load_LUT

.. autofunction:: ocpy.ls2.io.load_Kd_tables

Look-up Tables
^^^^^^^^^^^^^^

The LS2 LUTs are stored in ``ocpy/data/LS2/`` and contain pre-computed relationships
between Rrs and IOPs for various combinations of:

* Water types (oligotrophic to eutrophic)
* Solar zenith angles (0°, 30°, 60°)
* Viewing geometries

Loading the LUTs:

.. code-block:: python

   from ocpy.ls2.io import load_LUT, load_Kd_tables

   # Load main LS2 look-up tables
   LUT = load_LUT()

   # Load Kd neural network weights
   Kd_weights = load_Kd_tables()

Kd Neural Network
-----------------

.. module:: ocpy.ls2.kd_nn
   :synopsis: Neural network for Kd estimation

The Kd neural network provides estimates of the diffuse attenuation coefficient (Kd)
from Rrs, which is required as input to the LS2 algorithm.

.. autofunction:: ocpy.ls2.kd_nn.load_weights

.. autofunction:: ocpy.ls2.kd_nn.Kd_NN_MODIS

.. autofunction:: ocpy.ls2.kd_nn.MLP_Kd

Neural Network Architecture
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Kd estimation uses a multi-layer perceptron (MLP) trained on radiative transfer
simulations. Two separate networks handle:

* **Clear waters**: Low chlorophyll, open ocean conditions
* **Turbid waters**: High scattering, coastal conditions

The appropriate network is selected automatically based on Rrs characteristics.

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from ocpy.ls2.kd_nn import Kd_NN_MODIS, load_weights

   # Load weights
   weights_clear = load_weights('clear')
   weights_turbid = load_weights('turbid')

   # Example MODIS wavelengths and Rrs
   wavelengths = np.array([412, 443, 488, 531, 551, 667])
   Rrs = np.array([0.005, 0.006, 0.007, 0.008, 0.007, 0.001])
   sza = 30.0

   # Estimate Kd
   Kd = Kd_NN_MODIS(Rrs, sza, wavelengths)
   print(f"Estimated Kd: {Kd}")

Theoretical Background
----------------------

The relationship between Rrs and IOPs is described by radiative transfer theory.
For a homogeneous water column:

.. math::

   R_{rs} \\approx \\frac{f}{Q} \\cdot \\frac{b_b}{a + b_b}

where:

* f/Q is a bidirectional factor depending on sun and viewing geometry
* bb is the total backscattering coefficient
* a is the total absorption coefficient

The LS2 model improves upon this simple relationship by:

1. Using LUTs derived from full radiative transfer simulations
2. Accounting for Raman scattering
3. Including sun angle dependencies

Validation
----------

The LS2 algorithm has been validated against:

* In-situ IOP measurements from BOUSSOLE, MOBY, and other programs
* Hydrolight radiative transfer simulations
* Other semi-analytical algorithms (QAA, GSM)

Typical uncertainties:

* **a(443)**: 15-25% in clear waters, 30-50% in turbid waters
* **bb(555)**: 20-30% across water types

References
----------

* Loisel, H., Stramski, D., Dessailly, D., Jamet, C., Li, L., and Reynolds, R.A. (2018).
  An inverse model for estimating the optical absorption and backscattering coefficients
  of seawater from remote-sensing reflectance over a broad range of oceanic and coastal
  marine environments. Journal of Geophysical Research: Oceans, 123, 2141-2171.

* Loisel, H. and Stramski, D. (2000). Estimation of the inherent optical properties of
  natural waters from the irradiance attenuation coefficient and reflectance in the
  presence of Raman scattering. Applied Optics, 39(18), 3001-3011.
