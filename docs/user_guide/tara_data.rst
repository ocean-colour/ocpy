===============
Tara Oceans Data
===============

This guide covers working with Tara Oceans expedition data in ocpy.

Introduction
------------

The Tara Oceans expedition (2009-2013) was a global survey of marine plankton
communities, collecting:

* Bio-optical measurements (absorption, scattering)
* Biogeochemical data (pigments, nutrients)
* Genomic data (metagenomics, metatranscriptomics)
* Physical oceanography (CTD profiles)

ocpy provides tools for loading and analyzing the bio-optical component of this dataset.

Data Setup
----------

Before using the Tara modules, you need to download the data files:

1. Check ``ocpy/data/Tara/README.md`` for download instructions
2. Place the following files in ``ocpy/data/Tara/``:

   * ``tara_pg.parquet`` - Patrick Gray database
   * ``tara_ac.feather`` - Alison Chase database (optional)

Loading the Database
--------------------

.. code-block:: python

   from oceancolor.tara import io as tara_io

   # Load Patrick Gray database (default)
   tara = tara_io.load_db(dataset='pg')

   print(f"Number of samples: {len(tara)}")
   print(f"Columns: {list(tara.columns)[:20]}...")  # First 20 columns

   # Load as GeoDataFrame for spatial analysis
   tara_geo = tara_io.load_pg_db(expedition='all', as_geo=True)

   # Load specific expedition
   tara_polar = tara_io.load_pg_db(expedition='polar')

Available Data Fields
^^^^^^^^^^^^^^^^^^^^^

The database includes:

**Metadata**

* ``station``, ``cruise``, ``date``
* ``lat``, ``lon``, ``depth``
* ``ocean``, ``biome`` (Longhurst province)

**Physical Parameters**

* ``sst``, ``sss`` (sea surface temperature/salinity)
* CTD profile parameters

**Optical Properties**

* ``ap_*``: Particulate absorption at various wavelengths
* ``aph_*``: Phytoplankton absorption
* ``ad_*``: Detrital absorption
* ``ag_*``: CDOM absorption
* ``bp_*``: Particulate scattering
* ``bbp_*``: Particulate backscattering

Working with Spectra
--------------------

Extracting Spectra
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from oceancolor.tara import io as tara_io
   from oceancolor.tara import spectra

   # Load database
   tara = tara_io.load_db()

   # Extract particulate absorption spectra
   wavelengths, ap_data = spectra.spectra_from_table(tara, flavor='ap')
   print(f"Wavelengths: {wavelengths}")
   print(f"Spectra shape: {ap_data.shape}")  # (n_samples, n_wavelengths)

   # Extract phytoplankton absorption
   wavelengths, aph_data = spectra.spectra_from_table(tara, flavor='aph')

   # Extract CDOM absorption
   wavelengths, ag_data = spectra.spectra_from_table(tara, flavor='ag')

Single Station Spectra
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from oceancolor.tara import spectra

   # Get spectrum for a single row
   row = tara.iloc[0]  # First sample

   wave, ap = spectra.spectrum_from_row(row, flavor='ap')
   wave, aph = spectra.spectrum_from_row(row, flavor='aph')

   # Plot the spectra
   import matplotlib.pyplot as plt

   fig, ax = plt.subplots(figsize=(10, 6))
   ax.plot(wave, ap, 'b-', linewidth=2, label='a_p (total particulate)')
   ax.plot(wave, aph, 'g-', linewidth=2, label='a_ph (phytoplankton)')
   ax.set_xlabel('Wavelength (nm)')
   ax.set_ylabel('Absorption (m⁻¹)')
   ax.set_title(f"Station: {row['station']}")
   ax.legend()
   ax.grid(True, alpha=0.3)

Average Spectra
^^^^^^^^^^^^^^^

.. code-block:: python

   from oceancolor.tara import spectra

   # Calculate average spectrum
   wave, ap_mean = spectra.average_spectrum(tara, flavor='ap')

   # Calculate average for a subset
   tara_pacific = tara[tara['ocean'] == 'Pacific']
   wave, ap_pacific = spectra.average_spectrum(tara_pacific, flavor='ap')

Single Wavelength Values
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from oceancolor.tara import spectra

   # Extract value at specific wavelength
   ap_443 = spectra.single_value(tara, wv_cen=443, flavor='ap')
   print(f"a_p(443) range: {ap_443.min():.4f} - {ap_443.max():.4f} m⁻¹")

Derived Quantities
------------------

.. code-block:: python

   from oceancolor.tara import measures

   # Add derived quantities to the database
   tara = measures.add_derived(tara, quantities=['chl', 'poc'])

   # Calculate chlorophyll using BOSS method
   chl = measures.chla_boss13(tara)
   print(f"Chlorophyll range: {chl.min():.2f} - {chl.max():.2f} mg/m³")

   # Calculate particulate organic carbon
   poc = measures.poc(tara)
   print(f"POC range: {poc.min():.1f} - {poc.max():.1f} mg/m³")

Spatial Analysis
----------------

Distance to Coast
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from oceancolor.tara import analysis

   # Calculate distance to coast for all stations
   distances = analysis.dist_coast()
   print(f"Distance range: {distances.min():.0f} - {distances.max():.0f} km")

Mapping Stations
^^^^^^^^^^^^^^^^

.. code-block:: python

   import matplotlib.pyplot as plt
   import cartopy.crs as ccrs
   import cartopy.feature as cfeature
   import numpy as np

   def plot_tara_stations(tara, color_by='chl', title='Tara Oceans Stations'):
       """Plot Tara stations on a map."""
       fig, ax = plt.subplots(
           figsize=(14, 8),
           subplot_kw={'projection': ccrs.Robinson()}
       )

       ax.add_feature(cfeature.LAND, facecolor='lightgray')
       ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
       ax.gridlines(draw_labels=False, alpha=0.3)

       # Color by specified variable
       if color_by in tara.columns:
           c = tara[color_by]
           if color_by in ['chl', 'poc']:
               norm = plt.matplotlib.colors.LogNorm(
                   vmin=np.nanpercentile(c, 5),
                   vmax=np.nanpercentile(c, 95)
               )
           else:
               norm = None

           scatter = ax.scatter(
               tara['lon'], tara['lat'],
               c=c, s=30, alpha=0.7,
               transform=ccrs.PlateCarree(),
               cmap='viridis', norm=norm
           )
           plt.colorbar(scatter, ax=ax, label=color_by, shrink=0.6)
       else:
           ax.scatter(
               tara['lon'], tara['lat'],
               s=30, alpha=0.7,
               transform=ccrs.PlateCarree()
           )

       ax.set_title(title)
       return fig, ax

   # Plot stations colored by chlorophyll
   # plot_tara_stations(tara, color_by='chl')

Exploratory Analysis
--------------------

UMAP Projections
^^^^^^^^^^^^^^^^

.. code-block:: python

   from oceancolor.tara import io as tara_io

   # Load pre-computed UMAP projection
   umap_coords = tara_io.load_tara_umap(utype='aph')
   print(f"UMAP shape: {umap_coords.shape}")

Clustering
^^^^^^^^^^

.. code-block:: python

   from oceancolor.tara import explore

   # Prepare spectra for analysis
   wavelengths, spectra_clean = explore.prep_spectra(
       wv_grid=None,  # Use default grid
       min_sn=1.0     # Minimum signal-to-noise
   )

   # Run clustering analysis
   # clusters = explore.run_sequencer(wavelengths, spectra_clean)

Spectral Analysis Examples
--------------------------

Comparing Ocean Basins
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np
   from oceancolor.tara import io as tara_io
   from oceancolor.tara import spectra

   tara = tara_io.load_db()

   # Get unique oceans
   oceans = tara['ocean'].unique()

   fig, ax = plt.subplots(figsize=(10, 6))

   for ocean in oceans:
       subset = tara[tara['ocean'] == ocean]
       if len(subset) > 10:  # Minimum samples
           wave, ap_mean = spectra.average_spectrum(subset, flavor='ap')
           ax.plot(wave, ap_mean, label=f'{ocean} (n={len(subset)})')

   ax.set_xlabel('Wavelength (nm)')
   ax.set_ylabel('Mean a_p (m⁻¹)')
   ax.set_title('Particulate Absorption by Ocean Basin')
   ax.legend()
   ax.grid(True, alpha=0.3)

IOP Relationships
^^^^^^^^^^^^^^^^^

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np
   from oceancolor.tara import io as tara_io
   from oceancolor.tara import spectra

   tara = tara_io.load_db()

   # Get single-wavelength values
   ap_443 = spectra.single_value(tara, wv_cen=443, flavor='ap')
   aph_443 = spectra.single_value(tara, wv_cen=443, flavor='aph')
   bbp_443 = spectra.single_value(tara, wv_cen=443, flavor='bbp')

   # Create scatter plots
   fig, axes = plt.subplots(1, 2, figsize=(12, 5))

   # a_ph vs a_p
   ax = axes[0]
   valid = ~np.isnan(ap_443) & ~np.isnan(aph_443)
   ax.scatter(ap_443[valid], aph_443[valid], alpha=0.5, s=20)
   ax.plot([0, 0.5], [0, 0.5], 'k--', label='1:1')
   ax.set_xlabel('a_p(443) (m⁻¹)')
   ax.set_ylabel('a_ph(443) (m⁻¹)')
   ax.set_title('Phytoplankton vs Total Particulate Absorption')
   ax.legend()

   # bb_p vs a_p
   ax = axes[1]
   valid = ~np.isnan(ap_443) & ~np.isnan(bbp_443)
   ax.scatter(ap_443[valid], bbp_443[valid], alpha=0.5, s=20)
   ax.set_xlabel('a_p(443) (m⁻¹)')
   ax.set_ylabel('bb_p(443) (m⁻¹)')
   ax.set_title('Backscattering vs Absorption')

   plt.tight_layout()

Spectral Slope Analysis
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from scipy import stats

   def calc_spectral_slope(wavelengths, spectrum, wave_range=(400, 500)):
       """Calculate spectral slope using linear regression in log space."""
       idx = (wavelengths >= wave_range[0]) & (wavelengths <= wave_range[1])
       wave_subset = wavelengths[idx]
       spec_subset = spectrum[idx]

       # Remove zeros/negatives
       valid = spec_subset > 0
       if np.sum(valid) < 3:
           return np.nan

       log_spec = np.log(spec_subset[valid])
       wave_valid = wave_subset[valid]

       slope, intercept, r, p, se = stats.linregress(wave_valid, log_spec)
       return -slope  # Convention: positive slope for decreasing spectrum

   # Calculate spectral slopes for all samples
   wavelengths, ap_spectra = spectra.spectra_from_table(tara, flavor='ap')

   slopes = []
   for i in range(len(ap_spectra)):
       slope = calc_spectral_slope(wavelengths, ap_spectra[i])
       slopes.append(slope)

   slopes = np.array(slopes)
   print(f"Mean spectral slope: {np.nanmean(slopes):.4f} nm⁻¹")
   print(f"Slope range: {np.nanpercentile(slopes, 10):.4f} - {np.nanpercentile(slopes, 90):.4f} nm⁻¹")

Data Quality
------------

.. code-block:: python

   import numpy as np
   from oceancolor.tara import io as tara_io
   from oceancolor.tara import spectra

   tara = tara_io.load_db()

   # Check data completeness
   wavelengths, ap_data = spectra.spectra_from_table(tara, flavor='ap')

   # Count valid spectra
   valid_spectra = np.sum(~np.any(np.isnan(ap_data), axis=1))
   print(f"Valid a_p spectra: {valid_spectra}/{len(ap_data)}")

   # Check for negative values
   negative_count = np.sum(np.any(ap_data < 0, axis=1))
   print(f"Spectra with negative values: {negative_count}")

   # Summary statistics by wavelength
   for i, wl in enumerate(wavelengths[::10]):  # Every 10th wavelength
       valid = ~np.isnan(ap_data[:, i*10])
       print(f"a_p({wl:.0f}): n={np.sum(valid)}, "
             f"mean={np.nanmean(ap_data[:, i*10]):.4f}")

References
----------

* Boss, E., et al. (2013). The characteristics of particulate absorption, scattering
  and attenuation coefficients in the surface ocean. Methods in Oceanography, 7, 52-62.

* Bricaud, A., et al. (2012). Optical classification of Tara Oceans waters.
  Remote Sensing of Environment, 123, 509-523.

* Tara Oceans Foundation: https://fondationtaraocean.org/en/home/

* Tara Oceans data portal: https://www.ebi.ac.uk/metagenomics/studies/MGYS00002008
