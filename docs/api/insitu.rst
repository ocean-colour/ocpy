==============
In situ data
==============

.. module:: ocpy.insitu
   :synopsis: Loaders for in situ bio-optical compilations

The ``insitu`` subpackage provides loaders for community in situ bio-optical
compilations. The PANGAEA (Valente et al. 2022) version-3 dataset is the
best-supported; see the :doc:`PANGAEA user guide <../pangaea>` for a
narrative introduction.

PANGAEA
-------

.. module:: ocpy.insitu.pangaea
   :synopsis: PANGAEA V3 bio-optical in situ dataset

.. autofunction:: ocpy.insitu.pangaea.file_catalog

.. autofunction:: ocpy.insitu.pangaea.pangaea_path

.. autofunction:: ocpy.insitu.pangaea.dataset_file

.. autofunction:: ocpy.insitu.pangaea.find_header_end

.. autofunction:: ocpy.insitu.pangaea.load

.. autofunction:: ocpy.insitu.pangaea.spectrum

.. autofunction:: ocpy.insitu.pangaea.to_long

.. autofunction:: ocpy.insitu.pangaea.n_spectral

.. autofunction:: ocpy.insitu.pangaea.extract_hyperspectral

.. autofunction:: ocpy.insitu.pangaea.column_metadata

GLORIA
------

.. module:: ocpy.insitu.gloria
   :synopsis: GLORIA hyperspectral Rrs dataset

.. autofunction:: ocpy.insitu.gloria.load_gloria
