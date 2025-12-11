"""
Hyper-a Library of Core Functions.

Computation of absorption follows procedure described in:

Röttgers, R., W. Schönfeld, P.-R. Kipp, and R. Doerffer, 2005: Practical test of
a point-source integrating cavity absorption meter: the performance of different
collector assemblies. Appl. Opt., 44: 5549–5560.

Sequoia Scientific, Inc.
Python port v2.0
"""

import warnings
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.signal import savgol_filter

from .io import HyperaData, HyperaCalibration, HyperaConfig


# Constants for record IDs
NO_FILTER_RECORD_ID = 10
DARK_RECORD_ID = 999
CHLA_FILTER_RECORD_IDS = [601]
CHLA_SPF_WAVELENGTH_RANGE = (305, 595)

# IOCCG 2018 pure water absorption data
_IOCCG_AW_DATA = {
    'wl': np.array([
        250, 255, 260, 265, 270, 275, 280, 285, 290, 295, 300, 305, 310, 315, 320,
        325, 330, 335, 340, 345, 350, 355, 360, 365, 370, 375, 380, 385, 390, 395,
        400, 405, 410, 415, 420, 425, 430, 435, 440, 445, 450, 455, 460, 465, 470,
        475, 480, 485, 490, 495, 500, 505, 510, 515, 520, 525, 530, 535, 540, 545,
        550, 555, 560, 565, 570, 575, 580, 585, 590, 595, 600, 605, 610, 615, 620,
        625, 630, 635, 640, 645, 650, 655, 660, 665, 670, 675, 680, 685, 690, 695,
        700, 705, 710, 715, 720, 725, 730, 735, 740, 745, 750, 755, 760, 765, 770,
        775, 780, 785, 790, 795, 800, 805, 810, 815, 820, 825, 830, 835, 840, 845,
        850, 855, 860, 865, 870, 875, 880, 885, 890, 895, 900
    ], dtype=float),
    'aw': np.array([
        0.0450, 0.0392, 0.0344, 0.0303, 0.0269, 0.0240, 0.0216, 0.0194, 0.0176,
        0.0160, 0.0147, 0.0134, 0.0124, 0.0114, 0.0106, 0.0098, 0.0092, 0.0085,
        0.0080, 0.0075, 0.0071, 0.0068, 0.0066, 0.0063, 0.0060, 0.0056, 0.0052,
        0.0050, 0.0048, 0.0047, 0.0046, 0.0046, 0.0046, 0.0046, 0.00454, 0.00478,
        0.00495, 0.00530, 0.00635, 0.00751, 0.00922, 0.00962, 0.00979, 0.01011,
        0.0106, 0.0114, 0.0127, 0.0136, 0.0150, 0.0173, 0.0204, 0.0256, 0.0325,
        0.0396, 0.0409, 0.0417, 0.0434, 0.0452, 0.0474, 0.0511, 0.0565, 0.0596,
        0.0619, 0.0642, 0.0695, 0.0772, 0.0896, 0.110, 0.1351, 0.1672, 0.2224,
        0.2577, 0.2644, 0.2678, 0.2755, 0.2834, 0.2916, 0.3012, 0.3108, 0.3250,
        0.340, 0.371, 0.410, 0.429, 0.439, 0.448, 0.465, 0.486, 0.516, 0.559,
        0.624, 0.704, 0.827, 1.007, 1.231, 1.489, 1.970, 2.510, 2.780, 2.830,
        2.850, 2.880, 2.860, 2.860, 2.820, 2.760, 2.690, 2.590, 2.470, 2.360,
        2.250, 2.200, 2.190, 2.230, 2.340, 2.610, 3.220, 3.720, 3.940, 4.090,
        4.200, 4.320, 4.600, 4.600, 4.770, 5.010, 5.280, 5.570, 5.850, 6.130, 6.400
    ]),
    'psi_t': 1e-4 * np.array([
        3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, -1, -1, 0, 0,
        -0.7, -1, 0, 0, 0, -0.2, 0.1, 0.1, 0, 0.2, 0, -0.1, -0.1, 0, 0, 0.1, 0.2,
        0.1, 0.1, 0, -0.1, -0.1, 0, -0.1, -0.1, 0, 0.1, 0.3, 0.8, 1.2, 1.1, 0.7,
        0.4, 0.1, 0, -0.1, 0, -0.2, -0.4, -0.6, -0.7, -0.6, 0, 1.2, 2.5, 4.5, 7.9,
        10.3, 9.5, 7.2, 5.4, 3, 0.9, -0.5, -2, -3, -3, -2.2, 0.8, 0.9, -0.4, -2.1,
        -4, -4, -4.3, -4.1, -2, 1.7, 11, 27.3, 46.3, 65.8, 98.3, 148.5, 161, 137.2,
        105, 74.4, 44.7, 18.6, -4.4, -24.5, -40.4, -52.1, -59.4, -62, -60, -52,
        -38.4, -20, 0, 33, 101, 153, 145, 115, 83, 49, 27, -0.8, -46, -63, -78, -87,
        -90, -85, -70
    ]),
    'psi_s': 1e-4 * np.array([
        0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43,
        0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43,
        0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.43, 0.37, 0.36, 0.34, 0.32, 0.28,
        0.26, 0.25, 0.22, 0.19, 0.17, 0.16, 0.14, 0.13, 0.11, 0.09, 0.08, 0.06,
        0.06, 0.05, 0.05, 0.04, 0.02, 0.08, 0.13, 0.14, 0.15, 0.15, 0.15, 0.13,
        0.13, 0.16, 0.16, 0.15, 0.13, 0.10, 0.04, 0.01, 0.03, -0.02, -0.16, 0.43,
        0.75, 0.83, 0.84, 0.80, 0.77, 0.73, 0.70, 0.66, 0.60, 0.34, 0.41, 0.63,
        0.62, 0.46, 0.25, -0.02, -0.34, -0.70, -1.16, -1.40, -1.80, -1.90, -1.80,
        -2.10, -4.40, -3.70, 1.80, 4.70, 6.50, 6.80, 6.60, 6.00, 5.30, 4.50, 3.60,
        2.30, 1.10, -0.10, -1.30, -2.80, -4.40, -5.20, -6.20, -9.00, -13.0, -8.0,
        -2.0, 0, 0, 0, -2, -6, -8, -12, -16, -20, -22, -23, -25
    ])
}


def get_ioccg_aw(wls: np.ndarray, T: float, S: float) -> np.ndarray:
    """
    Compute theoretical pure water absorption using IOCCG 2018 data.

    Parameters
    ----------
    wls : np.ndarray
        Wavelengths (nm)
    T : float
        Temperature (Celsius)
    S : float
        Salinity (PSU)

    Returns
    -------
    np.ndarray
        Pure water absorption coefficient (1/m) at specified wavelengths,
        corrected for temperature and salinity
    """
    if wls is None or len(wls) == 0:
        wls = _IOCCG_AW_DATA['wl']

    wls = np.atleast_1d(wls)

    # Interpolate base values
    aw = np.interp(wls, _IOCCG_AW_DATA['wl'], _IOCCG_AW_DATA['aw'])
    psi_t = np.interp(wls, _IOCCG_AW_DATA['wl'], _IOCCG_AW_DATA['psi_t'])
    psi_s = np.interp(wls, _IOCCG_AW_DATA['wl'], _IOCCG_AW_DATA['psi_s'])

    # Apply temperature and salinity corrections
    # Reference: T=22°C, S=0 PSU
    aw_ts = aw + (T - 22) * psi_t + (S - 0) * psi_s

    return aw_ts


def ps(a: Union[float, np.ndarray], r: float) -> Union[float, np.ndarray]:
    """
    Probability function for integrating sphere.

    Röttgers et al. 2005 - Equation 5

    Parameters
    ----------
    a : float or np.ndarray
        Absorption coefficient (1/m)
    r : float
        Sphere radius (m)

    Returns
    -------
    float or np.ndarray
        Probability function value
    """
    ar = a * r
    with np.errstate(divide='ignore', invalid='ignore'):
        result = (1 - np.exp(-2 * ar) * (2 * ar + 1)) / (2 * a**2 * r**2)
        # Handle a=0 case
        if np.isscalar(a):
            if a == 0:
                result = 1.0
        else:
            result = np.where(a == 0, 1.0, result)
    return result


def compute_transmission(
    sample_data: pd.DataFrame,
    purewater_data: pd.DataFrame,
    config: HyperaConfig,
    f_fluor: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute transmission I/I0.

    Parameters
    ----------
    sample_data : pd.DataFrame
        Sample measurement data
    purewater_data : pd.DataFrame
        Pure water reference data
    config : HyperaConfig
        Instrument configuration
    f_fluor : np.ndarray, optional
        Chlorophyll fluorescence correction

    Returns
    -------
    np.ndarray
        Transmission array (n_samples x n_wavelengths)
    """
    sample_sig = np.vstack(sample_data['sig_pix'].values).astype(float)
    sample_ref = np.vstack(sample_data['ref_pix'].values).astype(float)
    purewater_sig = np.vstack(purewater_data['sig_pix'].values).astype(float)
    purewater_ref = np.vstack(purewater_data['ref_pix'].values).astype(float)

    # Apply chlorophyll fluorescence correction if provided
    if f_fluor is not None:
        no_filter_idx = sample_data['record_id'].values == NO_FILTER_RECORD_ID
        sample_sig[no_filter_idx, :] = sample_sig[no_filter_idx, :] - f_fluor

    # Reference-corrected signals
    with np.errstate(divide='ignore', invalid='ignore'):
        sample_ref_corr = sample_sig / sample_ref
        purewater_ref_corr = purewater_sig / purewater_ref

    # Compute transmission for each record ID
    T_AB = np.full((len(sample_data), sample_sig.shape[1]), np.nan)

    for record_id in sample_data['record_id'].unique():
        sample_idx = sample_data['record_id'].values == record_id
        purewater_idx = purewater_data['record_id'].values == record_id

        if not np.any(purewater_idx):
            continue

        # Median purewater for this ID
        purewater_ref_corr_median = np.nanmedian(purewater_ref_corr[purewater_idx, :], axis=0)

        T_AB[sample_idx, :] = sample_ref_corr[sample_idx, :] / purewater_ref_corr_median

    return T_AB


def compute_absorption(
    cal: HyperaCalibration,
    aw_ref: np.ndarray,
    T_AB: np.ndarray
) -> np.ndarray:
    """
    Compute absorption from Hyper-a data given a known reference.

    Parameters
    ----------
    cal : HyperaCalibration
        Calibration data
    aw_ref : np.ndarray
        Reference water absorption (1/m)
    T_AB : np.ndarray
        Transmission array (n_samples x n_wavelengths)

    Returns
    -------
    np.ndarray
        Total absorption coefficient (1/m)
    """
    r = cal.r
    r_0 = cal.r_0
    rho = cal.rho

    # Smooth transmission data
    if T_AB.shape[0] > 1:
        T_AB_smooth = np.apply_along_axis(
            lambda x: savgol_filter(x, min(15, len(x) // 2 * 2 + 1), 3)
            if len(x) > 15 else x,
            axis=1, arr=T_AB
        )
    else:
        T_AB_smooth = T_AB.copy()
        if T_AB_smooth.shape[1] > 15:
            T_AB_smooth[0, :] = savgol_filter(T_AB_smooth[0, :], 15, 3)

    a_tot_hypera = np.full(T_AB.shape, np.nan)

    for i_wl in range(len(aw_ref)):
        aw_i = aw_ref[i_wl]
        rho_i = rho[i_wl]

        def t_ab_model(a_guess):
            """Model for transmission given absorption guess."""
            ps_ref = ps(aw_i, r)
            ps_guess = ps(a_guess, r)
            return (
                np.exp(-r_0 * (a_guess - aw_i)) *
                ((1 - rho_i * ps_ref) / (1 - rho_i * ps_guess)) *
                (ps_guess / ps_ref)
            )

        for i_meas in range(T_AB.shape[0]):
            t_measured = T_AB_smooth[i_meas, i_wl]

            if np.isnan(t_measured) or t_measured <= 0:
                continue

            def objective(a_guess):
                """Minimize absolute percent error."""
                t_model = t_ab_model(a_guess)
                return abs(t_model - t_measured) / t_measured

            # Use bounded minimization starting from reference absorption
            try:
                result = minimize_scalar(
                    objective,
                    bounds=(0, 100),
                    method='bounded',
                    options={'xatol': 1e-8}
                )
                a_tot_hypera[i_meas, i_wl] = result.x
            except Exception:
                a_tot_hypera[i_meas, i_wl] = np.nan

    return a_tot_hypera


def compute_rho(
    cal: HyperaCalibration,
    a_A_known: np.ndarray,
    a_B_known: np.ndarray,
    T_AB: np.ndarray
) -> np.ndarray:
    """
    Compute sphere reflectivity given two solutions with known absorption.

    Röttgers et al. 2005 - Equation 10

    Parameters
    ----------
    cal : HyperaCalibration
        Calibration data
    a_A_known : np.ndarray
        Known absorption of solution A (e.g., ND spot + water)
    a_B_known : np.ndarray
        Known absorption of solution B (e.g., pure water)
    T_AB : np.ndarray
        Measured transmission between solutions A and B

    Returns
    -------
    np.ndarray
        Computed sphere reflectivity
    """
    r = cal.r
    r_0 = cal.r_0

    # Smooth transmission
    if T_AB.ndim == 1:
        if len(T_AB) > 15:
            T_AB = savgol_filter(T_AB, 15, 3)
    else:
        T_AB = np.apply_along_axis(
            lambda x: savgol_filter(x, min(15, len(x) // 2 * 2 + 1), 3)
            if len(x) > 15 else x,
            axis=1, arr=T_AB
        )
        T_AB = T_AB.flatten() if T_AB.shape[0] == 1 else np.median(T_AB, axis=0)

    ps_A = ps(a_A_known, r)
    ps_B = ps(a_B_known, r)

    exp_A = np.exp(-a_A_known * r_0)
    exp_B = np.exp(-a_B_known * r_0)

    # Röttgers et al. 2005 - Equation 10
    numerator = T_AB * exp_B * ps_B - exp_A * ps_A
    denominator = T_AB * exp_B * ps_A * ps_B - exp_A * ps_B * ps_A

    with np.errstate(divide='ignore', invalid='ignore'):
        rho = numerator / denominator

    return rho


def linearity_correct_pixels(data: HyperaData) -> HyperaData:
    """
    Apply spectrometer linearity correction.

    Parameters
    ----------
    data : HyperaData
        Data with sig_pix and ref_pix columns

    Returns
    -------
    HyperaData
        Data with linearity-corrected pixel values
    """
    config = data.config

    if config.sig_spec_lin_coeff is None or config.ref_spec_lin_coeff is None:
        warnings.warn('No linearity coefficients found in configuration, '
                      'linearity correction not applied.')
        return data

    df = data.data.copy()

    sig_pix = np.vstack(df['sig_pix'].values).astype(float)
    ref_pix = np.vstack(df['ref_pix'].values).astype(float)

    # Apply polynomial correction (coefficients are in MATLAB order - flip for numpy)
    sig_coeff = config.sig_spec_lin_coeff[::-1]
    ref_coeff = config.ref_spec_lin_coeff[::-1]

    sig_pix_corr = sig_pix / np.polyval(sig_coeff, sig_pix)
    ref_pix_corr = ref_pix / np.polyval(ref_coeff, ref_pix)

    df['sig_pix'] = list(sig_pix_corr)
    df['ref_pix'] = list(ref_pix_corr)

    return HyperaData(config=config, data=df)


def dark_correct_spectrum(data: HyperaData) -> HyperaData:
    """
    Remove dark measurements from Hyper-a spectrometer data.

    Parameters
    ----------
    data : HyperaData
        Data including dark measurements

    Returns
    -------
    HyperaData
        Dark-corrected data with dark records removed
    """
    df = data.data.copy()

    dark_idx = df['record_id'] == DARK_RECORD_ID
    dark_data = df[dark_idx].copy()
    df = df[~dark_idx].copy()

    if len(dark_data) == 0:
        warnings.warn('No dark measurements found in data.')
        return HyperaData(config=data.config, data=df)

    dark_sig = np.vstack(dark_data['sig_pix'].values).astype(float)
    dark_ref = np.vstack(dark_data['ref_pix'].values).astype(float)
    dark_dates = dark_data['date'].values

    sig_pix = np.vstack(df['sig_pix'].values).astype(float)
    ref_pix = np.vstack(df['ref_pix'].values).astype(float)

    for i in range(len(df)):
        # Find closest dark measurement in time
        sample_date = df['date'].iloc[i]
        if sample_date is not None and not pd.isna(sample_date):
            time_diffs = np.abs([(sample_date - d).total_seconds() if d is not None else np.inf
                                 for d in dark_dates])
            closest_idx = np.argmin(time_diffs)
        else:
            closest_idx = 0

        sig_pix[i, :] = sig_pix[i, :] - dark_sig[closest_idx, :]
        ref_pix[i, :] = ref_pix[i, :] - dark_ref[closest_idx, :]

    df['sig_pix'] = list(sig_pix)
    df['ref_pix'] = list(ref_pix)

    return HyperaData(config=data.config, data=df)


def interpolate_pixels_to_cal_wls(cal: HyperaCalibration, data: HyperaData) -> HyperaData:
    """
    Interpolate pixel values onto calibration wavelengths.

    Parameters
    ----------
    cal : HyperaCalibration
        Calibration with target wavelengths
    data : HyperaData
        Data with sig_pix and ref_pix columns

    Returns
    -------
    HyperaData
        Data with interpolated pixel values
    """
    config = data.config
    df = data.data.copy()

    sig_wls = config.sig_wls
    ref_wls = config.ref_wls
    cal_wls = cal.wl

    sig_pix = np.vstack(df['sig_pix'].values).astype(float)
    ref_pix = np.vstack(df['ref_pix'].values).astype(float)

    # Interpolate each row
    sig_pix_interp = np.zeros((sig_pix.shape[0], len(cal_wls)))
    ref_pix_interp = np.zeros((ref_pix.shape[0], len(cal_wls)))

    for i in range(sig_pix.shape[0]):
        sig_pix_interp[i, :] = np.interp(cal_wls, sig_wls, sig_pix[i, :])
        ref_pix_interp[i, :] = np.interp(cal_wls, ref_wls, ref_pix[i, :])

    df['sig_pix'] = list(sig_pix_interp)
    df['ref_pix'] = list(ref_pix_interp)

    return HyperaData(config=config, data=df)


def get_median_of_filter_runs(
    data: HyperaData,
    record_ids_to_median: list
) -> HyperaData:
    """
    Calculate median for specified record IDs, keep original data for others.

    Parameters
    ----------
    data : HyperaData
        Input data
    record_ids_to_median : list
        List of record IDs that should be median-binned

    Returns
    -------
    HyperaData
        Data with consecutive runs of specified IDs replaced by their median
    """
    df = data.data.copy()

    if len(df) == 0:
        return data

    # Identify consecutive runs based on record_id changes
    run_id = (df['record_id'] != df['record_id'].shift()).cumsum()

    results = []
    for _, run_data in df.groupby(run_id, sort=False):
        if run_data['record_id'].iloc[0] in record_ids_to_median:
            # Calculate median for this run
            median_row = {}
            for col in run_data.columns:
                if col in ['sig_pix', 'ref_pix']:
                    pix_data = np.vstack(run_data[col].values)
                    median_row[col] = np.median(pix_data, axis=0)
                elif col == 'date':
                    # Take middle date
                    median_row[col] = run_data[col].iloc[len(run_data) // 2]
                elif np.issubdtype(run_data[col].dtype, np.number):
                    median_row[col] = run_data[col].median()
                else:
                    median_row[col] = run_data[col].iloc[0]
            results.append(pd.DataFrame([median_row]))
        else:
            results.append(run_data)

    if results:
        df_result = pd.concat(results, ignore_index=True)
    else:
        df_result = df

    return HyperaData(config=data.config, data=df_result)


def compute_chl_fluorescence_correction(
    cal: HyperaCalibration,
    sample: HyperaData,
    purewater: HyperaData
) -> np.ndarray:
    """
    Calculate chlorophyll fluorescence correction (IOCCG 2018).

    Parameters
    ----------
    cal : HyperaCalibration
        Calibration data
    sample : HyperaData
        Sample measurements
    purewater : HyperaData
        Pure water reference measurements

    Returns
    -------
    np.ndarray
        Fluorescence correction array (n_no_filter_samples x n_wavelengths)
    """
    sample_df = sample.data
    purewater_df = purewater.data
    wl = cal.wl

    # Get indices for different measurement types
    pw_no_filter_idx = purewater_df['record_id'] == NO_FILTER_RECORD_ID
    pw_spf_idx = purewater_df['record_id'].isin(CHLA_FILTER_RECORD_IDS)

    sample_no_filter_idx = sample_df['record_id'] == NO_FILTER_RECORD_ID
    sample_spf_idx = sample_df['record_id'].isin(CHLA_FILTER_RECORD_IDS)

    # Pure water signals
    pw_no_filter_sig = np.vstack(purewater_df.loc[pw_no_filter_idx, 'sig_pix'].values)
    pw_no_filter_ref = np.vstack(purewater_df.loc[pw_no_filter_idx, 'ref_pix'].values)
    pw_spf_sig = np.vstack(purewater_df.loc[pw_spf_idx, 'sig_pix'].values)
    pw_spf_ref = np.vstack(purewater_df.loc[pw_spf_idx, 'ref_pix'].values)

    pw_no_filter_sig_median = np.median(pw_no_filter_sig, axis=0)
    pw_no_filter_ref_median = np.median(pw_no_filter_ref, axis=0)
    pw_spf_sig_median = np.median(pw_spf_sig, axis=0)
    pw_spf_ref_median = np.median(pw_spf_ref, axis=0)

    # Sample signals
    sample_no_filter_df = sample_df[sample_no_filter_idx].copy()
    sample_no_filter_sig = np.vstack(sample_no_filter_df['sig_pix'].values)
    sample_no_filter_ref = np.vstack(sample_no_filter_df['ref_pix'].values)

    sample_spf_df = sample_df[sample_spf_idx].copy()
    sample_spf_sig = np.vstack(sample_spf_df['sig_pix'].values)
    sample_spf_ref = np.vstack(sample_spf_df['ref_pix'].values)

    # Wavelength mask for chlorophyll excitation range
    chla_excitation_wls = (wl > CHLA_SPF_WAVELENGTH_RANGE[0]) & (wl < CHLA_SPF_WAVELENGTH_RANGE[1])

    n_no_filter = len(sample_no_filter_df)
    f_fluor = np.zeros((n_no_filter, len(wl)))

    for n in range(n_no_filter):
        # Find closest SPF measurement in time
        sample_date = sample_no_filter_df['date'].iloc[n]
        if sample_date is not None and len(sample_spf_df) > 0:
            time_diffs = np.abs([
                (sample_date - d).total_seconds() if d is not None else np.inf
                for d in sample_spf_df['date'].values
            ])
            closest_idx = np.argmin(time_diffs)
        else:
            closest_idx = 0

        if len(sample_spf_df) == 0:
            continue

        sample_spf_sig_n = sample_spf_sig[closest_idx, :]
        sample_spf_ref_n = sample_spf_ref[closest_idx, :]

        # Calculate scaling factors
        with np.errstate(divide='ignore', invalid='ignore'):
            S = sample_no_filter_ref[n, :] / pw_no_filter_ref_median
            S_SPF = sample_spf_ref_n / pw_spf_ref_median

        S_SPF[~chla_excitation_wls] = 0

        # Compute total absorbed light
        a_no_filter = np.nansum(S * pw_no_filter_sig_median - sample_no_filter_sig[n, :])
        a_spf = np.nansum(
            S_SPF[chla_excitation_wls] * pw_spf_sig_median[chla_excitation_wls] -
            sample_spf_sig_n[chla_excitation_wls]
        )

        # R_f: Scaling factor (0.1 determined empirically for PSICAM)
        if a_spf != 0:
            R_f = (a_no_filter / a_spf) + 0.1
        else:
            R_f = 0.1

        # Reference-corrected transmission
        pw_no_filter_idx_bool = purewater_df['record_id'] == NO_FILTER_RECORD_ID
        pw_no_filter_data = purewater_df[pw_no_filter_idx_bool]
        pw_sig = np.vstack(pw_no_filter_data['sig_pix'].values)
        pw_ref = np.vstack(pw_no_filter_data['ref_pix'].values)

        with np.errstate(divide='ignore', invalid='ignore'):
            T_AB = ((sample_no_filter_sig[n, :] / sample_no_filter_ref[n, :]) /
                    np.median(pw_sig / pw_ref, axis=0))

        # Scale SPF measured fluorescence
        f_fluor[n, :] = R_f * (sample_spf_sig_n - pw_spf_sig_median * T_AB)

    # Zero out below 660nm
    f_fluor[:, wl < 660] = 0

    return f_fluor
