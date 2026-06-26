"""
Hyper-a: Hyperspectral Absorption Meter Processing

This module provides Python tools for processing data from the Hyper-a
integrating cavity absorption meter (Sequoia Scientific, Inc.).

Computation of absorption follows procedure described in:

Rottgers, R., W. Schonfeld, P.-R. Kipp, and R. Doerffer, 2005: Practical test of
a point-source integrating cavity absorption meter: the performance of different
collector assemblies. Appl. Opt., 44: 5549-5560.

Main Functions
--------------
process : Process raw Hyper-a data to absorption coefficient
rho_from_nd_spot : Compute new sphere reflectivity from ND spot measurement
process_with_variable_ts : Process with varying temperature/salinity

I/O Functions
-------------
read_bin : Read Hyper-a binary (.bin) files
load_calibration : Load calibration .mat files
import_hypera_data : Import data from file or variable

Data Structures
---------------
HyperaResult : Container for processing results
HyperaCalibration : Calibration data
HyperaData : Raw data container
HyperaConfig : Instrument configuration

Library Functions
-----------------
get_ioccg_aw : Compute pure water absorption (IOCCG 2018)
compute_absorption : Compute absorption from transmission
compute_transmission : Compute transmission I/I0
compute_rho : Compute sphere reflectivity

Examples
--------
Basic processing workflow:

>>> from ocpy.hyper_a import process, load_calibration
>>>
>>> # Load calibration
>>> cal = load_calibration('CAL_20240801.mat')
>>>
>>> # Process sample data
>>> result = process(
...     cal=cal,
...     purewater='PureWater.bin',
...     T_purewater=22,
...     S_purewater=0,
...     sample='Sample.bin',
...     T_sample=22,
...     S_sample=35
... )
>>>
>>> # Access results
>>> wavelengths = result.wavelengths  # nm
>>> absorption = result.absorption    # 1/m, water absorption removed

Correcting for cavity reflectivity drift:

>>> from ocpy.hyper_a import rho_from_nd_spot
>>>
>>> # Compute new rho from ND spot measurement
>>> new_rho, a_spot = rho_from_nd_spot(
...     cal=cal,
...     purewater='PureWater.bin',
...     T_purewater=22,
...     spot='NDSpot.bin',
...     T_spot=22
... )
>>>
>>> # Update calibration
>>> cal.rho = new_rho
>>>
>>> # Process with corrected rho
>>> result = process(cal, purewater, 22, 0, sample, 22, 35)
"""

from .io import (
    HyperaConfig,
    HyperaCalibration,
    HyperaData,
    read_bin,
    load_calibration,
    load_mat_data,
    import_hypera_data,
)

from .lib import (
    NO_FILTER_RECORD_ID,
    DARK_RECORD_ID,
    CHLA_FILTER_RECORD_IDS,
    CHLA_SPF_WAVELENGTH_RANGE,
    get_ioccg_aw,
    ps,
    compute_transmission,
    compute_absorption,
    compute_rho,
    linearity_correct_pixels,
    dark_correct_spectrum,
    interpolate_pixels_to_cal_wls,
    get_median_of_filter_runs,
    compute_chl_fluorescence_correction,
)

from .process import (
    HyperaResult,
    process,
    rho_from_nd_spot,
    process_with_variable_ts,
)

__all__ = [
    # Data structures
    'HyperaConfig',
    'HyperaCalibration',
    'HyperaData',
    'HyperaResult',
    # I/O functions
    'read_bin',
    'load_calibration',
    'load_mat_data',
    'import_hypera_data',
    # Main processing functions
    'process',
    'rho_from_nd_spot',
    'process_with_variable_ts',
    # Library functions
    'get_ioccg_aw',
    'ps',
    'compute_transmission',
    'compute_absorption',
    'compute_rho',
    'linearity_correct_pixels',
    'dark_correct_spectrum',
    'interpolate_pixels_to_cal_wls',
    'get_median_of_filter_runs',
    'compute_chl_fluorescence_correction',
    # Constants
    'NO_FILTER_RECORD_ID',
    'DARK_RECORD_ID',
    'CHLA_FILTER_RECORD_IDS',
    'CHLA_SPF_WAVELENGTH_RANGE',
]
