==========
Satellites
==========

.. module:: ocpy.satellites
   :synopsis: Satellite sensor utilities

The ``satellites`` module provides utilities for working with ocean color satellite
data from various sensors.

PACE
----

.. module:: ocpy.satellites.pace
   :synopsis: PACE satellite utilities

Functions for the Plankton, Aerosol, Cloud, ocean Ecosystem (PACE) mission.

.. autofunction:: ocpy.satellites.pace.wave

.. autofunction:: ocpy.satellites.pace.gen_noise_vector

PACE Overview
^^^^^^^^^^^^^

PACE is NASA's most advanced ocean color mission, featuring:

* **OCI (Ocean Color Instrument)**: Hyperspectral coverage 340-890 nm with 5 nm resolution
* **SPEXone**: Multi-angle polarimeter (385-770 nm)
* **HARP2**: Hyperangular polarimeter

The OCI provides unprecedented spectral resolution for ocean color applications.

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from ocpy.satellites import pace

   # Generate PACE OCI wavelength array
   wavelengths = pace.wave(wv_min=400, wv_max=700, step=5)
   print(f"PACE wavelengths: {wavelengths}")

   # Generate noise/uncertainty vector
   noise = pace.gen_noise_vector(wavelengths, include_sampling=True)

   # Plot Rrs with uncertainty
   import matplotlib.pyplot as plt

   Rrs = np.array([...])  # Your Rrs spectrum
   plt.fill_between(wavelengths, Rrs - noise, Rrs + noise, alpha=0.3)
   plt.plot(wavelengths, Rrs)
   plt.xlabel('Wavelength (nm)')
   plt.ylabel('Rrs (sr⁻¹)')

MODIS
-----

.. module:: ocpy.satellites.modis
   :synopsis: MODIS satellite utilities

Functions for the Moderate Resolution Imaging Spectroradiometer (MODIS).

.. autofunction:: ocpy.satellites.modis.load_matchups

.. autofunction:: ocpy.satellites.modis.calc_errors

MODIS Specifications
^^^^^^^^^^^^^^^^^^^^

MODIS ocean color bands:

.. list-table:: MODIS Ocean Color Bands
   :header-rows: 1
   :widths: 20 30 50

   * - Band
     - Wavelength (nm)
     - Primary Use
   * - 8
     - 412
     - CDOM, aerosols
   * - 9
     - 443
     - Chlorophyll-a blue peak
   * - 10
     - 488
     - Chlorophyll-a
   * - 11
     - 531
     - Chlorophyll-a, fluorescence
   * - 12
     - 551
     - Reference (green)
   * - 13
     - 667
     - Atmospheric correction
   * - 14
     - 678
     - Chlorophyll fluorescence
   * - 15
     - 748
     - Aerosol, atmospheric correction

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

   from ocpy.satellites import modis

   # Load in-situ matchup data
   matchups = modis.load_matchups()
   print(f"Number of matchups: {len(matchups)}")

   # Calculate error statistics
   errors = modis.calc_errors(rel_in_situ_error=0.05)
   print(f"Error statistics: {errors}")

SeaWiFS
-------

.. module:: ocpy.satellites.seawifs
   :synopsis: SeaWiFS satellite utilities

Functions for the Sea-viewing Wide Field-of-view Sensor (SeaWiFS).

.. autofunction:: ocpy.satellites.seawifs.load_matchups

.. autofunction:: ocpy.satellites.seawifs.calc_errors

SeaWiFS Legacy
^^^^^^^^^^^^^^

SeaWiFS (1997-2010) established the foundation for modern ocean color remote sensing.

SeaWiFS bands:

* 412 nm - CDOM absorption
* 443 nm - Chlorophyll-a blue
* 490 nm - Chlorophyll-a
* 510 nm - Chlorophyll-a
* 555 nm - Reference green
* 670 nm - Atmospheric correction
* 765 nm - Aerosols
* 865 nm - Aerosols

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

   from ocpy.satellites import seawifs

   # Load matchup data
   matchups = seawifs.load_matchups()

   # Calculate errors
   errors = seawifs.calc_errors()

SBG
---

.. module:: ocpy.satellites.sbg
   :synopsis: SBG satellite utilities

Functions for the Surface Biology and Geology (SBG) mission.

.. autofunction:: ocpy.satellites.sbg.gen_noise_vector

Utilities
---------

.. module:: ocpy.satellites.utils
   :synopsis: Satellite utility functions

Common utility functions for satellite data processing.

.. autofunction:: ocpy.satellites.utils.calc_stats

Error Characterization
^^^^^^^^^^^^^^^^^^^^^^

Understanding satellite uncertainties is crucial for algorithm development:

.. code-block:: python

   from ocpy.satellites import utils

   # Calculate statistics from matchup data
   stats = utils.calc_stats(
       tbl=matchup_table,
       wv=wavelength,
       key_roots=['Rrs'],
       rel_in_situ_error=0.05
   )

Sensor Cross-Calibration
------------------------

When working with multiple sensors, wavelength differences must be considered:

.. list-table:: Ocean Color Sensor Wavelengths
   :header-rows: 1
   :widths: 15 12 12 12 12 12 12

   * - Sensor
     - UV
     - Blue1
     - Blue2
     - Green
     - Red
     - NIR
   * - SeaWiFS
     - 412
     - 443
     - 490
     - 555
     - 670
     - 865
   * - MODIS
     - 412
     - 443
     - 488
     - 551
     - 667
     - 869
   * - VIIRS
     - 410
     - 443
     - 486
     - 551
     - 671
     - 862
   * - PACE
     - 412
     - 443
     - 490
     - 555
     - 670
     - 865

Band-Shifting
^^^^^^^^^^^^^

To compare data from different sensors, band-shifting may be needed:

.. code-block:: python

   from scipy.interpolate import interp1d
   import numpy as np

   def band_shift(Rrs_source, wave_source, wave_target):
       """Shift Rrs from source to target wavelengths."""
       f = interp1d(wave_source, Rrs_source, kind='linear',
                    bounds_error=False, fill_value='extrapolate')
       return f(wave_target)

   # Example: MODIS to SeaWiFS
   modis_wave = np.array([412, 443, 488, 531, 551, 667])
   seawifs_wave = np.array([412, 443, 490, 510, 555, 670])

   Rrs_seawifs = band_shift(Rrs_modis, modis_wave, seawifs_wave)

Quality Control
---------------

Satellite data requires careful quality control:

**Common Quality Flags**

* **LAND**: Pixel over land
* **CLOUD**: Cloud contamination
* **HIGLINT**: High sun glint
* **HILT**: High sensor tilt
* **STRAYLIGHT**: Stray light contamination
* **ATMFAIL**: Atmospheric correction failure
* **LOWLW**: Low water-leaving radiance
* **MAXAERITER**: Maximum aerosol iterations exceeded

**Recommended Processing**

.. code-block:: python

   def apply_quality_mask(Rrs, flags, mask_bits):
       """Apply quality flags to Rrs data."""
       import numpy as np

       # Create mask from flag bits
       mask = np.zeros_like(flags, dtype=bool)
       for bit in mask_bits:
           mask |= (flags & (1 << bit)) > 0

       # Apply mask
       Rrs_masked = np.ma.array(Rrs, mask=mask)
       return Rrs_masked

References
----------

* Werdell, P.J. and Bailey, S.W. (2005). An improved in-situ bio-optical data set
  for ocean color algorithm development and satellite data product validation.
  Remote Sensing of Environment, 98(1), 122-140.

* McClain, C.R., Feldman, G.C., and Hooker, S.B. (2004). An overview of the SeaWiFS
  project and strategies for producing a climate research quality global ocean
  bio-optical time series. Deep Sea Research Part II, 51(1-3), 5-42.

* Werdell, P.J., et al. (2019). The Plankton, Aerosol, Cloud, ocean Ecosystem
  mission: status, science, advances. Bulletin of the American Meteorological
  Society, 100(9), 1775-1794.
