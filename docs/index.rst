.. ocpy documentation master file

====================================
ocpy: Ocean Color Analysis in Python
====================================

**ocpy** is a comprehensive Python package for ocean color analysis, providing tools for
processing satellite remote sensing data, calculating inherent and apparent optical properties,
and analyzing phytoplankton and water constituents.

.. image:: https://readthedocs.org/projects/ocpy/badge/?version=latest
   :target: https://ocpy.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

Overview
--------

Ocean color remote sensing is a powerful tool for understanding marine ecosystems, carbon cycling,
and water quality on global scales. **ocpy** provides a unified Python interface for:

* **Water Optical Properties**: Pure seawater absorption and scattering coefficients using
  established models (Zhang et al. 2009, IOCCG 2018)

* **IOP Inversions**: Derive absorption and backscattering from remote sensing reflectance
  using the LS2 model (Loisel et al. 2018) and ZLee methods

* **Chlorophyll-a Algorithms**: Standard band-ratio algorithms (OC2, OC4) for chlorophyll estimation

* **Phytoplankton Analysis**: Absorption spectra, pigment fitting, and community composition

* **Satellite Data Processing**: Support for PACE, MODIS, and SeaWiFS data products

* **In-situ Data**: Tools for Tara Oceans expedition data and Hyper-a absorption meter processing

* **Radiative Transfer**: Interface to Loisel+2023 Hydrolight simulation datasets

Key Features
------------

* Modular design with clear separation between optical property domains
* Compatible with NumPy arrays, pandas DataFrames, and xarray Datasets
* Includes reference datasets and look-up tables for standard algorithms
* Well-documented with examples and tutorials
* Tested against reference implementations

Quick Example
-------------

.. code-block:: python

   import numpy as np
   from oceancolor.water import absorption, scattering
   from oceancolor.chl import band_ratios

   # Get pure water absorption at PACE wavelengths
   wavelengths = np.arange(400, 701, 5)  # nm
   a_w = absorption.a_water(wavelengths, data='IOCCG')

   # Calculate seawater scattering
   b_w = scattering.betasw_ZHH2009(wavelengths, Tc=20, theta=90, S=35)

   # Estimate chlorophyll from Rrs using OC4
   Rrs = np.array([0.005, 0.006, 0.007, 0.008, 0.006])  # Example Rrs values
   wave = np.array([443, 490, 510, 555, 670])
   chl = band_ratios.oc4(wave, Rrs)

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/water_optics
   user_guide/iop_inversions
   user_guide/chlorophyll
   user_guide/phytoplankton
   user_guide/satellites
   user_guide/visualization
   user_guide/tara_data
   user_guide/hydrolight
   user_guide/hypera
   panagea

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/water
   api/ls2
   api/chl
   api/iop
   api/ph
   api/satellites
   api/tara
   api/hydrolight
   api/hypera
   api/pace
   api/utils

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Citation
--------

If you use ocpy in your research, please cite:

.. code-block:: bibtex

   @software{ocpy,
     author = {Prochaska, J. Xavier and contributors},
     title = {ocpy: Ocean Color Analysis in Python},
     url = {https://github.com/ocean-colour/ocpy},
     version = {0.1},
   }

License
-------

ocpy is released under the BSD 3-Clause License. See the LICENSE file for details.

Acknowledgments
---------------

This package builds upon decades of ocean color research. We acknowledge the contributions of:

* NASA Ocean Biology Processing Group (OBPG)
* IOCCG (International Ocean Colour Coordinating Group)
* The developers of the LS2 model (Loisel et al.)
* The Tara Oceans consortium
* Sequoia Scientific for the Hyper-a instrument

Contact
-------

* **Author**: J. Xavier Prochaska (jxp@ucsc.edu)
* **Repository**: https://github.com/ocean-colour/ocpy
* **Issues**: https://github.com/ocean-colour/ocpy/issues
