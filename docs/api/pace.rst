===================
PACE Data Products
===================

.. module:: oceancolor.pace
   :synopsis: PACE satellite data I/O

The ``pace`` module provides functions for loading and processing PACE (Plankton,
Aerosol, Cloud, ocean Ecosystem) satellite data products.

Data I/O
--------

.. module:: oceancolor.pace.io
   :synopsis: PACE Level 2 data loading

.. autofunction:: oceancolor.pace.io.load_oci_l2

.. autofunction:: oceancolor.pace.io.load_oci_l2_spectrum

.. autofunction:: oceancolor.pace.io.load_oci_l2_spectrum_pixel

.. autofunction:: oceancolor.pace.io.load_iop_l2

PACE OCI Level 2 Products
-------------------------

The Ocean Color Instrument (OCI) provides hyperspectral ocean color data with
unprecedented spectral resolution.

**Remote Sensing Reflectance (Rrs)**

The primary product for ocean color applications:

.. code-block:: python

   from oceancolor.pace import io as pace_io

   # Load OCI L2 file
   Rrs, Rrs_unc, FLH = pace_io.load_oci_l2('PACE_OCI.20240401.L2_OC.nc')

   print(f"Rrs dimensions: {Rrs.dims}")
   print(f"Wavelengths: {Rrs.wavelength.values}")

**Fluorescence Line Height (FLH)**

Chlorophyll fluorescence product for phytoplankton biomass estimation.

**IOP Products**

Inherent optical property retrievals:

.. code-block:: python

   from oceancolor.pace import io as pace_io

   # Load IOP products
   iop_data = pace_io.load_iop_l2('PACE_OCI.20240401.L2_IOP.nc')

   # Available products:
   # a - total absorption
   # bb - total backscattering
   # aph - phytoplankton absorption
   # adg - detrital + CDOM absorption

Data Structure
--------------

PACE L2 data is provided as xarray Datasets with:

**Dimensions**

* ``number_of_lines``: Along-track pixels
* ``pixels_per_line``: Cross-track pixels
* ``wavelength``: Spectral bands (hyperspectral)

**Coordinates**

* ``latitude``: Geographic latitude
* ``longitude``: Geographic longitude
* ``wavelength``: Wavelength in nanometers

**Data Variables**

* ``Rrs``: Remote sensing reflectance (sr⁻¹)
* ``Rrs_unc``: Rrs uncertainty (sr⁻¹)
* ``l2_flags``: Quality flags

Example Workflow
----------------

**Fast single-spectrum extraction** (recommended for point lookups):

.. code-block:: python

   import matplotlib.pyplot as plt
   from oceancolor.pace import io as pace_io

   # Extract single spectrum at a location (fast - avoids loading full granule)
   lat_target, lon_target = 35.0, -70.0
   wls, rrs, rrs_unc, flag, pixel_coords = pace_io.load_oci_l2_spectrum(
       'PACE_OCI_L2.nc', lat_target, lon_target
   )
   ix, iy, actual_lat, actual_lon = pixel_coords

   # Plot spectrum with uncertainty
   fig, ax = plt.subplots(figsize=(10, 6))
   ax.plot(wls, rrs)
   ax.fill_between(wls, rrs - rrs_unc, rrs + rrs_unc, alpha=0.3)
   ax.set_xlabel('Wavelength (nm)')
   ax.set_ylabel('Rrs (sr⁻¹)')
   ax.set_title(f'PACE OCI Rrs at ({actual_lat:.2f}, {actual_lon:.2f})')

**Full granule analysis** (when processing many pixels):

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from oceancolor.pace import io as pace_io
   from oceancolor.chl import band_ratios
   from oceancolor.satellites import pace

   # Load full PACE granule
   xds, flags = pace_io.load_oci_l2('PACE_OCI_L2.nc')

   # Generate standard wavelength array
   wave = pace.wave(wv_min=400, wv_max=700, step=5)

   # Extract Rrs at a specific location
   lat_target, lon_target = 35.0, -70.0

   # Find nearest pixel
   lat_arr = xds.latitude.values
   lon_arr = xds.longitude.values
   dist = np.sqrt((lat_arr - lat_target)**2 + (lon_arr - lon_target)**2)
   row, col = np.unravel_index(np.argmin(dist), dist.shape)

   # Get spectrum at location
   Rrs_spectrum = xds.Rrs.values[row, col, :]
   Rrs_unc_spectrum = xds.Rrs_unc.values[row, col, :]
   wavelengths = xds.wavelength.values

   # Plot spectrum
   fig, ax = plt.subplots(figsize=(10, 6))
   ax.plot(wavelengths, Rrs_spectrum)
   ax.fill_between(
       wavelengths,
       Rrs_spectrum - Rrs_unc_spectrum,
       Rrs_spectrum + Rrs_unc_spectrum,
       alpha=0.3
   )
   ax.set_xlabel('Wavelength (nm)')
   ax.set_ylabel('Rrs (sr⁻¹)')
   ax.set_title(f'PACE OCI Rrs at ({lat_target}, {lon_target})')

Quality Flags
-------------

PACE L2 data includes quality flags for each pixel:

.. list-table:: PACE L2 Quality Flags
   :header-rows: 1
   :widths: 10 30 60

   * - Bit
     - Name
     - Description
   * - 0
     - ATMFAIL
     - Atmospheric correction failure
   * - 1
     - LAND
     - Land pixel
   * - 2
     - PRODWARN
     - Product warning
   * - 3
     - HIGLINT
     - High sun glint
   * - 4
     - HILT
     - High sensor tilt
   * - 5
     - HISATZEN
     - High satellite zenith angle
   * - 6
     - COASTZ
     - Coastal zone
   * - 7
     - STRAYLIGHT
     - Stray light contamination
   * - 8
     - CLDICE
     - Cloud/ice
   * - 9
     - COCCOLITH
     - Coccolithophore bloom
   * - 10
     - TURBIDW
     - Turbid water
   * - 11
     - HISOLZEN
     - High solar zenith angle
   * - 12
     - LOWLW
     - Low water-leaving radiance
   * - 13
     - CHLFAIL
     - Chlorophyll algorithm failure
   * - 14
     - NAVWARN
     - Navigation warning
   * - 15
     - ABSAER
     - Absorbing aerosols

Applying quality masks:

.. code-block:: python

   import numpy as np

   def apply_pace_qc(Rrs, flags, bad_flags=None):
       """Apply quality control flags to PACE Rrs."""
       if bad_flags is None:
           # Default bad flags
           bad_flags = [0, 1, 3, 4, 7, 8, 12]  # ATMFAIL, LAND, HIGLINT, etc.

       # Create mask
       mask = np.zeros_like(flags, dtype=bool)
       for bit in bad_flags:
           mask |= (flags & (1 << bit)) > 0

       # Apply mask
       Rrs_masked = Rrs.where(~mask)
       return Rrs_masked

   # Apply QC
   Rrs_clean = apply_pace_qc(Rrs, Rrs.l2_flags)

PACE Hyperspectral Analysis
---------------------------

PACE's hyperspectral capability enables advanced analysis:

**Derivative Spectroscopy**

.. code-block:: python

   import numpy as np

   def spectral_derivative(wavelengths, Rrs):
       """Calculate spectral derivative of Rrs."""
       dRrs = np.gradient(Rrs, wavelengths)
       return dRrs

   # Find absorption features
   dRrs = spectral_derivative(wavelengths, Rrs_spectrum.values)

   # Plot derivative
   fig, axes = plt.subplots(2, 1, figsize=(10, 8))
   axes[0].plot(wavelengths, Rrs_spectrum.values)
   axes[0].set_ylabel('Rrs (sr⁻¹)')
   axes[1].plot(wavelengths, dRrs)
   axes[1].set_xlabel('Wavelength (nm)')
   axes[1].set_ylabel('dRrs/dλ')

**Phytoplankton Community Detection**

.. code-block:: python

   # Look for diagnostic wavelengths
   # Cyanobacteria: absorption near 620 nm (phycocyanin)
   # Coccolithophores: high reflectance, low Rrs slope

   def detect_cyano(wavelengths, Rrs):
       """Detect cyanobacteria from PACE Rrs."""
       # Simple ratio indicator
       idx_620 = np.argmin(np.abs(wavelengths - 620))
       idx_665 = np.argmin(np.abs(wavelengths - 665))
       idx_681 = np.argmin(np.abs(wavelengths - 681))

       # Phycocyanin absorption creates trough near 620 nm
       cyano_index = (Rrs[idx_620] - 0.5*(Rrs[idx_665] + Rrs[idx_681])) / Rrs[idx_681]
       return cyano_index

Data Access
-----------

PACE data is available from NASA Earthdata:

* **Ocean Color Web**: https://oceancolor.gsfc.nasa.gov/
* **Earthdata Search**: https://search.earthdata.nasa.gov/

File naming convention:

.. code-block:: text

   PACE_OCI.YYYYMMDDTHHMMSS.L2_OC.V2.0.NRT.nc

* YYYY: Year
* MM: Month
* DD: Day
* HHMMSS: Time (UTC)
* L2_OC: Level 2 Ocean Color
* V2.0: Processing version
* NRT: Near Real-Time (or blank for reprocessed)

References
----------

* Werdell, P.J., et al. (2019). The Plankton, Aerosol, Cloud, ocean Ecosystem
  mission: status, science, advances. Bulletin of the American Meteorological
  Society, 100(9), 1775-1794.

* NASA PACE Mission: https://pace.gsfc.nasa.gov/

* PACE Algorithm Theoretical Basis Documents:
  https://pace.oceansciences.org/docs.htm
