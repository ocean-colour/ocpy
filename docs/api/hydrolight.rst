======================
Hydrolight Simulations
======================

.. module:: oceancolor.hydrolight
   :synopsis: Hydrolight radiative transfer data

The ``hydrolight`` module provides access to Hydrolight radiative transfer simulation
datasets, particularly the Loisel+2023 compilation.

Loisel+2023 Dataset
-------------------

.. module:: oceancolor.hydrolight.loisel23
   :synopsis: Loisel+2023 Hydrolight datasets

The Loisel et al. (2023) dataset contains extensive Hydrolight simulations spanning
a wide range of oceanic conditions.

.. autofunction:: oceancolor.hydrolight.loisel23.load_ds

.. autofunction:: oceancolor.hydrolight.loisel23.calc_Chl

Dataset Organization
^^^^^^^^^^^^^^^^^^^^

The dataset is organized by:

* **X**: Inelastic scattering scenario

  * X=1: No inelastic scattering
  * X=2: Raman scattering only
  * X=4: Raman + chlorophyll fluorescence

* **Y**: Solar zenith angle

  * Y=0: Sun at zenith (0°)
  * Y=30: 30° solar zenith angle
  * Y=60: 60° solar zenith angle

Setup Requirements
^^^^^^^^^^^^^^^^^^

The dataset requires setting the ``OS_COLOR`` environment variable:

.. code-block:: bash

   # Download from Dryad: https://doi.org/10.6076/D1630T
   export OS_COLOR="/path/to/loisel2023/data"

Or in Python:

.. code-block:: python

   import os
   os.environ['OS_COLOR'] = '/path/to/loisel2023/data'

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

   import os
   os.environ['OS_COLOR'] = '/path/to/data'

   from oceancolor.hydrolight import loisel23

   # Load dataset with Raman scattering, 30° solar zenith
   ds = loisel23.load_ds(X=2, Y=30)

   # Examine dataset structure
   print(ds)
   print(f"Variables: {list(ds.data_vars)}")
   print(f"Dimensions: {dict(ds.dims)}")

   # Access Rrs
   Rrs = ds['Rrs']
   wavelength = ds['wavelength'].values

   # Calculate chlorophyll from simulation inputs
   chl = loisel23.calc_Chl(ds)
   print(f"Chlorophyll range: {chl.min().values:.2f} - {chl.max().values:.2f} mg/m³")

Dataset Variables
^^^^^^^^^^^^^^^^^

The xarray Dataset contains:

**Coordinates**

* ``wavelength``: Wavelengths in nm
* ``sample``: Simulation sample index

**Primary Variables**

* ``Rrs``: Remote sensing reflectance (sr⁻¹)
* ``Kd``: Diffuse attenuation coefficient (m⁻¹)
* ``a``: Total absorption coefficient (m⁻¹)
* ``bb``: Total backscattering coefficient (m⁻¹)

**IOP Components**

* ``a_w``: Pure water absorption
* ``a_ph``: Phytoplankton absorption
* ``a_dg``: Detrital + CDOM absorption
* ``bb_w``: Pure water backscattering
* ``bb_p``: Particulate backscattering

**Metadata**

* ``sza``: Solar zenith angle
* ``chl``: Chlorophyll-a concentration

Working with the Data
^^^^^^^^^^^^^^^^^^^^^

**Selecting Subsets**

.. code-block:: python

   # Select specific wavelength
   Rrs_443 = ds['Rrs'].sel(wavelength=443, method='nearest')

   # Select wavelength range
   Rrs_vis = ds['Rrs'].sel(wavelength=slice(400, 700))

   # Filter by chlorophyll
   chl = loisel23.calc_Chl(ds)
   low_chl = ds.where(chl < 0.1, drop=True)

**Statistical Analysis**

.. code-block:: python

   import numpy as np

   # Mean spectrum
   Rrs_mean = ds['Rrs'].mean(dim='sample')

   # Correlation between Rrs and Chl
   chl = loisel23.calc_Chl(ds)
   Rrs_443 = ds['Rrs'].sel(wavelength=443, method='nearest')
   correlation = np.corrcoef(Rrs_443.values.flatten(), chl.values.flatten())[0, 1]

**Plotting**

.. code-block:: python

   import matplotlib.pyplot as plt

   fig, axes = plt.subplots(1, 2, figsize=(12, 5))

   # Plot Rrs spectra
   ax = axes[0]
   for i in range(min(100, ds.dims['sample'])):
       ax.plot(ds['wavelength'], ds['Rrs'].isel(sample=i),
               alpha=0.1, color='blue')
   ax.set_xlabel('Wavelength (nm)')
   ax.set_ylabel('Rrs (sr⁻¹)')
   ax.set_title('Rrs Spectra')

   # Plot Rrs vs Chl at 443 nm
   ax = axes[1]
   chl = loisel23.calc_Chl(ds)
   Rrs_443 = ds['Rrs'].sel(wavelength=443, method='nearest')
   ax.scatter(chl, Rrs_443, alpha=0.3, s=5)
   ax.set_xscale('log')
   ax.set_xlabel('Chlorophyll (mg/m³)')
   ax.set_ylabel('Rrs(443) (sr⁻¹)')
   ax.set_title('Rrs(443) vs Chlorophyll')

   plt.tight_layout()

Simulation Parameters
---------------------

The Loisel+2023 simulations cover:

**Water Types**

* Oligotrophic (Chl < 0.1 mg/m³)
* Mesotrophic (0.1 < Chl < 1 mg/m³)
* Eutrophic (Chl > 1 mg/m³)
* Coastal (high CDOM and/or sediments)

**IOP Variability**

* Chlorophyll: 0.01 - 100 mg/m³
* CDOM: Variable S_CDOM (0.012 - 0.022 nm⁻¹)
* Particles: Variable size distributions

**Physical Conditions**

* Solar zenith angles: 0°, 30°, 60°
* Wind speed effects on sea surface
* Bottom reflectance (for shallow cases)

Use Cases
---------

**Algorithm Development**

Use the simulations to develop and test retrieval algorithms:

.. code-block:: python

   from oceancolor.hydrolight import loisel23
   from oceancolor.chl import band_ratios
   import numpy as np

   # Load simulations
   ds = loisel23.load_ds(X=2, Y=30)

   # Extract data for algorithm testing
   wave = ds['wavelength'].values
   Rrs = ds['Rrs'].values
   chl_true = loisel23.calc_Chl(ds).values

   # Test OC4 algorithm
   chl_oc4 = []
   for i in range(len(chl_true)):
       chl_oc4.append(band_ratios.oc4(wave, Rrs[i, :]))
   chl_oc4 = np.array(chl_oc4)

   # Calculate error statistics
   log_error = np.log10(chl_oc4) - np.log10(chl_true)
   bias = np.mean(log_error)
   rmse = np.sqrt(np.mean(log_error**2))
   print(f"Bias: {bias:.3f}, RMSE: {rmse:.3f}")

**Uncertainty Propagation**

Study how IOP uncertainties propagate to Rrs:

.. code-block:: python

   # Compare scenarios with/without Raman
   ds_no_raman = loisel23.load_ds(X=1, Y=30)
   ds_raman = loisel23.load_ds(X=2, Y=30)

   # Raman contribution
   Rrs_diff = ds_raman['Rrs'] - ds_no_raman['Rrs']
   raman_contribution = Rrs_diff / ds_raman['Rrs'] * 100  # Percent

References
----------

* Loisel, H., et al. (2023). A comprehensive database of in-water IOPs and
  water-leaving reflectances for benchmarking and validation of ocean color
  algorithms. Dryad Dataset. https://doi.org/10.6076/D1630T

* Mobley, C.D. (1994). Light and Water: Radiative Transfer in Natural Waters.
  Academic Press.

* Mobley, C.D. and Sundman, L.K. (2008). Hydrolight 5.0 Users' Guide.
  Sequoia Scientific.
