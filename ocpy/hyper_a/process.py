"""
Hyper-a Processing Functions.

Processes raw Hyper-a data to absorption coefficient (1/m). Data can be
imported from Hyper-a .bin file or .mat files.

Sequoia Scientific, Inc.
Python port v2.0
"""

from typing import Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .io import (
    HyperaData, HyperaCalibration, HyperaConfig,
    import_hypera_data, load_calibration, read_bin
)
from .lib import (
    NO_FILTER_RECORD_ID, DARK_RECORD_ID, CHLA_FILTER_RECORD_IDS,
    get_ioccg_aw, compute_transmission, compute_absorption, compute_rho,
    linearity_correct_pixels, dark_correct_spectrum,
    interpolate_pixels_to_cal_wls, get_median_of_filter_runs,
    compute_chl_fluorescence_correction
)


@dataclass
class HyperaResult:
    """Container for Hyper-a processing results."""
    absorption: np.ndarray  # Absorption coefficient (1/m), water absorption removed
    wavelengths: np.ndarray  # Wavelengths (nm)
    date: list  # Measurement timestamps
    depth: np.ndarray  # Depth (m)
    water_temp: np.ndarray  # Water temperature (C) from instrument
    input_voltage: np.ndarray  # Input voltage (V)
    calibration: HyperaCalibration  # Calibration used
    transmission: Optional[np.ndarray] = None  # Intermediate transmission values


def process(
    cal: Union[str, HyperaCalibration],
    purewater: Union[str, HyperaData],
    T_purewater: float,
    S_purewater: float,
    sample: Union[str, HyperaData],
    T_sample: float,
    S_sample: float,
    remove_water_absorption: bool = True,
    chl_fluor_corr: bool = True,
    return_transmission: bool = False
) -> Union[HyperaResult, Tuple[HyperaResult, np.ndarray]]:
    """
    Process raw Hyper-a data to absorption coefficient.

    Parameters
    ----------
    cal : str or HyperaCalibration
        Path to calibration .mat file or HyperaCalibration object
    purewater : str or HyperaData
        Path to pure water measurement file (.bin or .mat) or HyperaData object
    T_purewater : float
        Water temperature (Celsius) of the pure water measurement
    S_purewater : float
        Salinity (PSU) of the pure water measurement (typically 0)
    sample : str or HyperaData
        Path to sample measurement file (.bin or .mat) or HyperaData object
    T_sample : float
        Water temperature (Celsius) of the sample measurement
    S_sample : float
        Salinity (PSU) of the sample measurement
    remove_water_absorption : bool, optional
        Whether to subtract pure water absorption from result. Default True.
    chl_fluor_corr : bool, optional
        Whether to apply chlorophyll fluorescence correction. Default True.
    return_transmission : bool, optional
        Whether to return transmission as second output. Default False.

    Returns
    -------
    HyperaResult
        Processing results including absorption coefficients
    np.ndarray, optional
        Transmission array if return_transmission=True

    Notes
    -----
    The absorption coefficient in the result has water absorption removed
    (if remove_water_absorption=True).

    Examples
    --------
    >>> from ocpy.hyper_a import process, load_calibration
    >>> cal = load_calibration('CAL_20240801.mat')
    >>> result = process(
    ...     cal=cal,
    ...     purewater='PureWater.bin',
    ...     T_purewater=22,
    ...     S_purewater=0,
    ...     sample='Sample.bin',
    ...     T_sample=22,
    ...     S_sample=35
    ... )
    >>> print(result.absorption.shape)
    """
    # Load calibration if path provided
    if isinstance(cal, str):
        cal = load_calibration(cal)

    # Import data
    sample_data = import_hypera_data(sample)
    purewater_data = import_hypera_data(purewater)

    # Apply linearity correction
    sample_data = linearity_correct_pixels(sample_data)
    purewater_data = linearity_correct_pixels(purewater_data)

    # Median bin dark and chlorophyll short-pass filter measurements
    record_ids_to_median = [DARK_RECORD_ID] + CHLA_FILTER_RECORD_IDS
    sample_data = get_median_of_filter_runs(sample_data, record_ids_to_median)
    purewater_data = get_median_of_filter_runs(purewater_data, record_ids_to_median)

    # Subtract dark
    sample_data = dark_correct_spectrum(sample_data)
    purewater_data = dark_correct_spectrum(purewater_data)

    # Interpolate onto calibration wavelengths
    sample_data = interpolate_pixels_to_cal_wls(cal, sample_data)
    purewater_data = interpolate_pixels_to_cal_wls(cal, purewater_data)

    # Compute chlorophyll fluorescence correction
    f_fluor = None
    if chl_fluor_corr:
        f_fluor = compute_chl_fluorescence_correction(cal, sample_data, purewater_data)

    # Calculate transmission
    T_AB = compute_transmission(
        sample_data.data, purewater_data.data,
        sample_data.config, f_fluor
    )

    # Compute absorption with pure water as reference
    aw_ref = get_ioccg_aw(cal.wl, T_purewater, S_purewater)
    a_hypera = compute_absorption(cal, aw_ref, T_AB)

    # Remove absorption by water
    if remove_water_absorption:
        aw_sample = get_ioccg_aw(cal.wl, T_sample, S_sample)
        a_hypera = a_hypera - aw_sample

    # Keep only no-filter data
    no_filter_idx = sample_data.data['record_id'] == NO_FILTER_RECORD_ID
    T_AB_filtered = T_AB[no_filter_idx, :]
    a_hypera_filtered = a_hypera[no_filter_idx, :]
    sample_df_filtered = sample_data.data[no_filter_idx].copy()

    # Build result
    result = HyperaResult(
        absorption=a_hypera_filtered,
        wavelengths=cal.wl,
        date=sample_df_filtered['date'].tolist(),
        depth=sample_df_filtered['depth'].values,
        water_temp=sample_df_filtered['water_temp'].values,
        input_voltage=sample_df_filtered['input_voltage'].values,
        calibration=cal,
        transmission=T_AB_filtered if return_transmission else None
    )

    if return_transmission:
        return result, T_AB_filtered
    return result


def rho_from_nd_spot(
    cal: Union[str, HyperaCalibration],
    purewater: Union[str, HyperaData],
    T_purewater: float,
    spot: Union[str, HyperaData],
    T_spot: float
) -> Tuple[np.ndarray, HyperaResult]:
    """
    Compute new sphere reflectivity from a measurement of clean water and ND spot.

    This function can be used to correct for drift in cavity reflectivity.

    Parameters
    ----------
    cal : str or HyperaCalibration
        Path to calibration .mat file or HyperaCalibration object
    purewater : str or HyperaData
        Path to pure water measurement file with Fluorilon white plug installed
    T_purewater : float
        Water temperature (Celsius) of the pure water measurement
    spot : str or HyperaData
        Path to measurement file with ND black spot installed
    T_spot : float
        Water temperature (Celsius) of the spot measurement

    Returns
    -------
    rho : np.ndarray
        New sphere reflectivity based on the spot measurement
    a_spot : HyperaResult
        Spot absorption result (water absorption removed)

    Notes
    -----
    After computing the new rho, you can update the calibration:
    >>> new_rho, a_spot = rho_from_nd_spot(cal, purewater, 22, spot, 22)
    >>> cal.rho = new_rho
    >>> result = process(cal, purewater, 22, 0, sample, 22, 35)

    Examples
    --------
    >>> from ocpy.hyper_a import load_calibration, rho_from_nd_spot
    >>> cal = load_calibration('CAL_20240801.mat')
    >>> new_rho, a_spot = rho_from_nd_spot(
    ...     cal=cal,
    ...     purewater='PureWater.bin',
    ...     T_purewater=22,
    ...     spot='NDSpot.bin',
    ...     T_spot=22
    ... )
    >>> # Update calibration with new rho
    >>> cal.rho = new_rho
    """
    # Load calibration if path provided
    if isinstance(cal, str):
        cal = load_calibration(cal)

    # Process spot measurement (salinity assumed to be 0 PSU)
    a_spot_result, T_AB = process(
        cal, purewater, T_purewater, 0,
        spot, T_spot, 0,
        chl_fluor_corr=False,
        return_transmission=True
    )

    # Compute pure water absorption
    purewater_aw_modeled = get_ioccg_aw(cal.wl, T_purewater, 0)
    cal_spot_aw_modeled = get_ioccg_aw(cal.wl, T_spot, 0)

    # Add water absorption to known spot absorption
    if cal.spot_absorp is None:
        raise ValueError("Calibration file does not contain spot absorption data (spotAbsorp)")

    cal_spot_a_total = cal.spot_absorp + cal_spot_aw_modeled

    # Calculate new rho using known pure water and spot absorption values
    rho = compute_rho(
        cal,
        cal_spot_a_total,
        purewater_aw_modeled,
        np.median(T_AB, axis=0)
    )

    # Smooth the result
    if len(rho) > 30:
        rho = savgol_filter(rho, 30, 3)

    return rho, a_spot_result


def process_with_variable_ts(
    cal: Union[str, HyperaCalibration],
    purewater: Union[str, HyperaData],
    T_purewater: float,
    S_purewater: float,
    sample: Union[str, HyperaData],
    T_sample_array: np.ndarray,
    S_sample_array: np.ndarray,
    sample_timestamps: np.ndarray,
    chl_fluor_corr: bool = True
) -> HyperaResult:
    """
    Process Hyper-a data with varying temperature and salinity.

    Use this function when you have arrays of temperature and salinity
    measurements that correspond to different times during the sample
    collection.

    Parameters
    ----------
    cal : str or HyperaCalibration
        Path to calibration .mat file or HyperaCalibration object
    purewater : str or HyperaData
        Path to pure water measurement file or HyperaData object
    T_purewater : float
        Water temperature (Celsius) of the pure water measurement
    S_purewater : float
        Salinity (PSU) of the pure water measurement
    sample : str or HyperaData
        Path to sample measurement file or HyperaData object
    T_sample_array : np.ndarray
        Array of temperature values
    S_sample_array : np.ndarray
        Array of salinity values
    sample_timestamps : np.ndarray
        Timestamps corresponding to T and S arrays
    chl_fluor_corr : bool, optional
        Whether to apply chlorophyll fluorescence correction. Default True.

    Returns
    -------
    HyperaResult
        Processing results with water absorption removed using
        interpolated T/S values

    Examples
    --------
    >>> import numpy as np
    >>> from ocpy.hyper_a import process_with_variable_ts
    >>> # Your CTD data
    >>> ctd_temps = np.array([20.1, 20.3, 20.5, 20.2])
    >>> ctd_sals = np.array([35.1, 35.0, 34.9, 35.0])
    >>> ctd_times = ...  # Your CTD timestamps
    >>> result = process_with_variable_ts(
    ...     cal='CAL_20240801.mat',
    ...     purewater='PureWater.bin',
    ...     T_purewater=22,
    ...     S_purewater=0,
    ...     sample='Sample.bin',
    ...     T_sample_array=ctd_temps,
    ...     S_sample_array=ctd_sals,
    ...     sample_timestamps=ctd_times
    ... )
    """
    # First process without water absorption removal
    result = process(
        cal, purewater, T_purewater, S_purewater,
        sample, 0, 0,  # Placeholder T/S
        remove_water_absorption=False,
        chl_fluor_corr=chl_fluor_corr
    )

    # Load calibration if needed
    if isinstance(cal, str):
        cal = load_calibration(cal)

    # Interpolate T and S onto Hyper-a timestamps
    result_timestamps = np.array([
        d.timestamp() if hasattr(d, 'timestamp') else 0
        for d in result.date
    ])
    sample_ts_numeric = np.array([
        t.timestamp() if hasattr(t, 'timestamp') else float(t)
        for t in sample_timestamps
    ])

    T_interp = np.interp(result_timestamps, sample_ts_numeric, T_sample_array)
    S_interp = np.interp(result_timestamps, sample_ts_numeric, S_sample_array)

    # Calculate water absorption for each sample
    aw_sample = np.zeros_like(result.absorption)
    for i in range(len(result.date)):
        aw_sample[i, :] = get_ioccg_aw(cal.wl, T_interp[i], S_interp[i])

    # Subtract water absorption
    result.absorption = result.absorption - aw_sample

    return result
