======================
Hydrolight Simulations
======================

This guide covers working with Hydrolight radiative transfer simulation data in ocpy.

Introduction
------------

Hydrolight is the gold standard for ocean radiative transfer simulations. ocpy provides
access to the Loisel+2023 comprehensive simulation database, which contains thousands
of simulated Rrs spectra across a wide range of water types.

This data is invaluable for:

* Algorithm development and testing
* Uncertainty analysis
* Understanding Rrs-IOP relationships
* Training machine learning models

Data Setup
----------

The Loisel+2023 data requires external download:

1. Download from Dryad: https://doi.org/10.6076/D1630T
2. Set the ``OS_COLOR`` environment variable:

.. code-block:: bash

   export OS_COLOR="/path/to/loisel2023/data"

Or in Python:

.. code-block:: python

   import os
   os.environ['OS_COLOR'] = '/path/to/loisel2023/data'

Loading Data
------------

.. code-block:: python

   import os
   os.environ['OS_COLOR'] = '/path/to/data'  # Set your path

   from ocpy.hydrolight import loisel23

   # Load dataset
   # X: inelastic scattering scenario
   #    1 = no inelastic
   #    2 = Raman scattering
   #    4 = Raman + chlorophyll fluorescence
   # Y: solar zenith angle (0, 30, or 60 degrees)

   ds = loisel23.load_ds(X=2, Y=30)

   print(ds)
   print(f"\nVariables: {list(ds.data_vars)}")
   print(f"Dimensions: {dict(ds.dims)}")

Dataset Structure
-----------------

The xarray Dataset contains:

.. code-block:: python

   # Coordinates
   wavelength = ds['wavelength']  # nm
   sample = ds.dims['sample']     # Number of simulations

   # Primary variables
   Rrs = ds['Rrs']                # Remote sensing reflectance (sr⁻¹)
   Kd = ds['Kd']                  # Diffuse attenuation (m⁻¹)

   # IOPs
   a = ds['a']                    # Total absorption (m⁻¹)
   bb = ds['bb']                  # Total backscattering (m⁻¹)

   # IOP components (if available)
   # a_w, a_ph, a_dg, bb_w, bb_p

   # Calculate chlorophyll
   chl = loisel23.calc_Chl(ds)

Exploring the Data
------------------

Spectral Range
^^^^^^^^^^^^^^

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np

   ds = loisel23.load_ds(X=2, Y=30)

   wavelength = ds['wavelength'].values
   Rrs = ds['Rrs'].values

   # Plot a sample of spectra
   fig, ax = plt.subplots(figsize=(10, 6))

   n_plot = min(100, len(Rrs))
   indices = np.random.choice(len(Rrs), n_plot, replace=False)

   for i in indices:
       ax.plot(wavelength, Rrs[i], alpha=0.2, color='blue')

   ax.set_xlabel('Wavelength (nm)')
   ax.set_ylabel('Rrs (sr⁻¹)')
   ax.set_title('Sample Rrs Spectra from Hydrolight Simulations')
   ax.grid(True, alpha=0.3)

   plt.tight_layout()

Parameter Distributions
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np
   from ocpy.hydrolight import loisel23

   ds = loisel23.load_ds(X=2, Y=30)
   chl = loisel23.calc_Chl(ds)

   fig, axes = plt.subplots(2, 2, figsize=(12, 10))

   # Chlorophyll distribution
   ax = axes[0, 0]
   ax.hist(chl.values, bins=50, edgecolor='black', alpha=0.7)
   ax.set_xlabel('Chlorophyll (mg/m³)')
   ax.set_ylabel('Count')
   ax.set_xscale('log')
   ax.set_title('Chlorophyll Distribution')

   # Rrs at 443 nm
   ax = axes[0, 1]
   Rrs_443 = ds['Rrs'].sel(wavelength=443, method='nearest').values
   ax.hist(Rrs_443, bins=50, edgecolor='black', alpha=0.7)
   ax.set_xlabel('Rrs(443) (sr⁻¹)')
   ax.set_ylabel('Count')
   ax.set_title('Rrs(443) Distribution')

   # a(443)
   if 'a' in ds.data_vars:
       ax = axes[1, 0]
       a_443 = ds['a'].sel(wavelength=443, method='nearest').values
       ax.hist(a_443, bins=50, edgecolor='black', alpha=0.7)
       ax.set_xlabel('a(443) (m⁻¹)')
       ax.set_ylabel('Count')
       ax.set_xscale('log')
       ax.set_title('Absorption Distribution')

   # bb(555)
   if 'bb' in ds.data_vars:
       ax = axes[1, 1]
       bb_555 = ds['bb'].sel(wavelength=555, method='nearest').values
       ax.hist(bb_555, bins=50, edgecolor='black', alpha=0.7)
       ax.set_xlabel('bb(555) (m⁻¹)')
       ax.set_ylabel('Count')
       ax.set_xscale('log')
       ax.set_title('Backscattering Distribution')

   plt.tight_layout()

Algorithm Testing
-----------------

Use simulations to test retrieval algorithms:

.. code-block:: python

   import numpy as np
   from ocpy.hydrolight import loisel23
   from ocpy.chl.band_ratios import oc4

   ds = loisel23.load_ds(X=2, Y=30)
   chl_true = loisel23.calc_Chl(ds).values

   wavelength = ds['wavelength'].values
   Rrs = ds['Rrs'].values

   # Test OC4 algorithm
   # Find indices for OC4 wavelengths
   oc4_waves = np.array([443, 490, 510, 555, 670])
   wave_idx = [np.argmin(np.abs(wavelength - w)) for w in oc4_waves]

   chl_oc4 = []
   for i in range(len(chl_true)):
       Rrs_oc4 = Rrs[i, wave_idx]
       try:
           chl_est = oc4(oc4_waves, Rrs_oc4)
           chl_oc4.append(chl_est)
       except:
           chl_oc4.append(np.nan)

   chl_oc4 = np.array(chl_oc4)

   # Calculate statistics
   valid = ~np.isnan(chl_oc4) & (chl_true > 0) & (chl_oc4 > 0)
   log_error = np.log10(chl_oc4[valid]) - np.log10(chl_true[valid])

   bias = np.mean(log_error)
   rmse = np.sqrt(np.mean(log_error**2))
   r = np.corrcoef(np.log10(chl_true[valid]), np.log10(chl_oc4[valid]))[0, 1]

   print(f"OC4 Performance:")
   print(f"  Bias (log10): {bias:.3f}")
   print(f"  RMSE (log10): {rmse:.3f}")
   print(f"  Correlation (r): {r:.3f}")

   # Validation plot
   import matplotlib.pyplot as plt

   fig, ax = plt.subplots(figsize=(8, 8))
   ax.scatter(chl_true[valid], chl_oc4[valid], alpha=0.3, s=5)
   ax.plot([0.01, 100], [0.01, 100], 'k--', label='1:1')
   ax.set_xscale('log')
   ax.set_yscale('log')
   ax.set_xlabel('True Chlorophyll (mg/m³)')
   ax.set_ylabel('OC4 Chlorophyll (mg/m³)')
   ax.set_title('OC4 Algorithm Validation')
   ax.legend()
   ax.grid(True, alpha=0.3)

Comparing Scenarios
-------------------

Study the effects of inelastic scattering:

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np
   from ocpy.hydrolight import loisel23

   # Load scenarios
   ds_no_inelastic = loisel23.load_ds(X=1, Y=30)
   ds_raman = loisel23.load_ds(X=2, Y=30)

   wavelength = ds_no_inelastic['wavelength'].values

   # Raman contribution
   Rrs_no_raman = ds_no_inelastic['Rrs'].mean(dim='sample').values
   Rrs_raman = ds_raman['Rrs'].mean(dim='sample').values

   Rrs_diff = Rrs_raman - Rrs_no_raman
   raman_pct = Rrs_diff / Rrs_raman * 100

   fig, axes = plt.subplots(1, 2, figsize=(14, 5))

   # Absolute difference
   ax = axes[0]
   ax.plot(wavelength, Rrs_no_raman, label='No inelastic')
   ax.plot(wavelength, Rrs_raman, label='With Raman')
   ax.set_xlabel('Wavelength (nm)')
   ax.set_ylabel('Mean Rrs (sr⁻¹)')
   ax.set_title('Mean Rrs Spectra')
   ax.legend()
   ax.grid(True, alpha=0.3)

   # Percent contribution
   ax = axes[1]
   ax.plot(wavelength, raman_pct)
   ax.set_xlabel('Wavelength (nm)')
   ax.set_ylabel('Raman Contribution (%)')
   ax.set_title('Raman Scattering Effect')
   ax.grid(True, alpha=0.3)
   ax.axhline(y=0, color='k', linestyle='--')

   plt.tight_layout()

Sun Angle Effects
^^^^^^^^^^^^^^^^^

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np
   from ocpy.hydrolight import loisel23

   fig, ax = plt.subplots(figsize=(10, 6))

   for sza in [0, 30, 60]:
       ds = loisel23.load_ds(X=2, Y=sza)
       Rrs_mean = ds['Rrs'].mean(dim='sample').values
       ax.plot(ds['wavelength'].values, Rrs_mean, label=f'SZA = {sza}°')

   ax.set_xlabel('Wavelength (nm)')
   ax.set_ylabel('Mean Rrs (sr⁻¹)')
   ax.set_title('Solar Zenith Angle Effect on Rrs')
   ax.legend()
   ax.grid(True, alpha=0.3)

Subsetting Data
---------------

Working with subsets for specific water types:

.. code-block:: python

   import numpy as np
   from ocpy.hydrolight import loisel23

   ds = loisel23.load_ds(X=2, Y=30)
   chl = loisel23.calc_Chl(ds)

   # Oligotrophic waters (Chl < 0.1 mg/m³)
   ds_oligo = ds.where(chl < 0.1, drop=True)
   print(f"Oligotrophic samples: {ds_oligo.dims['sample']}")

   # Mesotrophic waters (0.1 < Chl < 1 mg/m³)
   ds_meso = ds.where((chl >= 0.1) & (chl < 1), drop=True)
   print(f"Mesotrophic samples: {ds_meso.dims['sample']}")

   # Eutrophic waters (Chl > 1 mg/m³)
   ds_eutro = ds.where(chl >= 1, drop=True)
   print(f"Eutrophic samples: {ds_eutro.dims['sample']}")

Machine Learning Training
-------------------------

Use the simulations for training ML models:

.. code-block:: python

   import numpy as np
   from sklearn.model_selection import train_test_split
   from sklearn.ensemble import RandomForestRegressor
   from ocpy.hydrolight import loisel23

   ds = loisel23.load_ds(X=2, Y=30)
   chl = loisel23.calc_Chl(ds).values
   Rrs = ds['Rrs'].values

   # Prepare features (band ratios)
   wavelength = ds['wavelength'].values
   idx_443 = np.argmin(np.abs(wavelength - 443))
   idx_490 = np.argmin(np.abs(wavelength - 490))
   idx_555 = np.argmin(np.abs(wavelength - 555))

   X = np.column_stack([
       np.log10(Rrs[:, idx_443] / Rrs[:, idx_555]),
       np.log10(Rrs[:, idx_490] / Rrs[:, idx_555]),
       Rrs[:, idx_443],
       Rrs[:, idx_555]
   ])

   y = np.log10(chl)

   # Remove invalid data
   valid = ~np.any(np.isnan(X), axis=1) & ~np.isnan(y) & ~np.isinf(y)
   X = X[valid]
   y = y[valid]

   # Split data
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )

   # Train model
   rf = RandomForestRegressor(n_estimators=100, random_state=42)
   rf.fit(X_train, y_train)

   # Evaluate
   y_pred = rf.predict(X_test)
   rmse = np.sqrt(np.mean((y_pred - y_test)**2))
   print(f"Test RMSE (log10 Chl): {rmse:.3f}")

References
----------

* Loisel, H., et al. (2023). A comprehensive database of in-water IOPs and
  water-leaving reflectances. Dryad Dataset. https://doi.org/10.6076/D1630T

* Mobley, C.D. (1994). Light and Water: Radiative Transfer in Natural Waters.
  Academic Press.

* Mobley, C.D. and Sundman, L.K. (2008). Hydrolight 5.0 Users' Guide.
