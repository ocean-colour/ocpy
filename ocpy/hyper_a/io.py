"""
Hyper-a I/O functions for reading binary files and loading calibration/data files.

Reads the binary (.bin) file produced by the Hyper-a instrument and loads
calibration files.

Sequoia Scientific, Inc.
Python port v2.0
"""

import os
import struct
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Union, Dict, Any

import numpy as np
import pandas as pd
import scipy.io as sio


@dataclass
class HyperaConfig:
    """Configuration structure for Hyper-a instrument."""
    serial_number: int = 0
    firmware_ver: float = 0.0
    sig_spec_sn: int = 0
    sef_spec_sn: int = 0
    sig_num_wls: int = 0
    ref_num_wls: int = 0
    config_byte: int = 0
    main_board_rev: str = ''
    pump_flush_sec: int = 0
    sequence_interval_sec: int = 0
    burst_interval_min: int = 0
    sequences_per_burst: int = 0
    file_interval_hours: int = 0
    sig_spec_lin_coeff: Optional[np.ndarray] = None
    ref_spec_lin_coeff: Optional[np.ndarray] = None
    meas_name: list = field(default_factory=list)
    meas_id: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.uint16))
    sig_wls: np.ndarray = field(default_factory=lambda: np.array([]))
    ref_wls: np.ndarray = field(default_factory=lambda: np.array([]))
    auto_exposure: bool = False
    sig_lin_corr: bool = False
    ref_lin_corr: bool = False
    pump_enabled: bool = False
    auto_start: bool = False
    switch_start: bool = False


@dataclass
class HyperaCalibration:
    """Calibration data for Hyper-a instrument."""
    wl: np.ndarray  # wavelengths
    r: float  # sphere radius
    r_0: float  # distance from source to sphere wall
    rho: np.ndarray  # sphere reflectivity
    date: str = ''
    proc_ver: float = 1.0
    serial_number: int = 0
    spot_absorp: Optional[np.ndarray] = None  # ND spot absorption


@dataclass
class HyperaData:
    """Container for Hyper-a data and configuration."""
    config: HyperaConfig
    data: pd.DataFrame


def read_bin(filename: str) -> HyperaData:
    """
    Read a Hyper-a binary (.bin) file.

    Parameters
    ----------
    filename : str
        Path to the .bin file

    Returns
    -------
    HyperaData
        Object containing config and data DataFrame

    Notes
    -----
    The data DataFrame contains columns:
        - record_id: Measurement type identifier
        - date: Timestamp of measurement
        - input_voltage: Input voltage (V)
        - water_temp: Water temperature (C)
        - depth: Depth (m)
        - board_temp: Board temperature (C)
        - board_humid: Board humidity (%)
        - lamp_freq: Lamp frequency
        - sig_avg: Signal averaging count
        - ref_avg: Reference averaging count
        - sig_exp: Signal exposure time
        - ref_exp: Reference exposure time
        - sig_num_flash: Number of signal flashes
        - ref_num_flash: Number of reference flashes
        - sig_pix: Signal pixel values (2D array)
        - ref_pix: Reference pixel values (2D array)
    """
    config = HyperaConfig()

    with open(filename, 'rb') as f:
        # Get file size
        f.seek(0, 2)
        file_size_bytes = f.tell()
        f.seek(0)

        # Parse header
        config.serial_number = struct.unpack('<H', f.read(2))[0]
        config.firmware_ver = struct.unpack('<H', f.read(2))[0] / 100.0
        config.sig_spec_sn = struct.unpack('<I', f.read(4))[0]
        config.sef_spec_sn = struct.unpack('<I', f.read(4))[0]
        config.sig_num_wls = struct.unpack('<H', f.read(2))[0]
        config.ref_num_wls = struct.unpack('<H', f.read(2))[0]
        config.config_byte = struct.unpack('<B', f.read(1))[0]
        config.main_board_rev = f.read(1).decode('ascii', errors='ignore')
        config.pump_flush_sec = struct.unpack('<H', f.read(2))[0]

        if config.firmware_ver >= 1.07:
            config.sequence_interval_sec = struct.unpack('<H', f.read(2))[0]
            config.burst_interval_min = struct.unpack('<H', f.read(2))[0]
            config.sequences_per_burst = struct.unpack('<H', f.read(2))[0]

            if config.firmware_ver >= 1.09:
                config.file_interval_hours = struct.unpack('<H', f.read(2))[0]
                config.sig_spec_lin_coeff = np.array(struct.unpack('<8f', f.read(32)))
                config.ref_spec_lin_coeff = np.array(struct.unpack('<8f', f.read(32)))
            else:
                f.read(2)  # unassigned, used for memory alignment

        # Read measurement names (8 names, 16 chars each)
        config.meas_name = []
        for _ in range(8):
            name = f.read(16).decode('ascii', errors='ignore').strip('\x00').strip()
            config.meas_name.append(name)

        config.meas_id = np.array(struct.unpack('<8H', f.read(16)), dtype=np.uint16)

        # Read wavelengths
        config.sig_wls = np.array(
            struct.unpack(f'<{config.sig_num_wls}I', f.read(4 * config.sig_num_wls))
        ) / 1000.0
        config.ref_wls = np.array(
            struct.unpack(f'<{config.ref_num_wls}I', f.read(4 * config.ref_num_wls))
        ) / 1000.0

        # Parse config byte flags
        config.auto_exposure = bool(config.config_byte & 1)
        config.sig_lin_corr = bool(config.config_byte & 2)
        config.ref_lin_corr = bool(config.config_byte & 4)
        config.pump_enabled = bool(config.config_byte & 8)
        config.auto_start = bool(config.config_byte & 16)
        config.switch_start = bool(config.config_byte & 32)

        # Calculate number of data records
        header_pos = f.tell()
        data_bytes_in_file = file_size_bytes - header_pos
        data_record_bytes = 32 + config.sig_num_wls * 2 + config.ref_num_wls * 2
        num_data_records = data_bytes_in_file // data_record_bytes

        if data_bytes_in_file % data_record_bytes != 0:
            import warnings
            warnings.warn('File contains incomplete data records')

        # Preallocate arrays
        records = {
            'record_id': np.zeros(num_data_records, dtype=np.uint16),
            'date': [None] * num_data_records,
            'input_voltage': np.zeros(num_data_records),
            'water_temp': np.zeros(num_data_records),
            'depth': np.zeros(num_data_records),
            'board_temp': np.zeros(num_data_records),
            'board_humid': np.zeros(num_data_records),
            'lamp_freq': np.zeros(num_data_records),
            'sig_avg': np.zeros(num_data_records),
            'ref_avg': np.zeros(num_data_records),
            'sig_exp': np.zeros(num_data_records, dtype=np.uint32),
            'ref_exp': np.zeros(num_data_records, dtype=np.uint32),
            'sig_num_flash': np.zeros(num_data_records, dtype=np.uint16),
            'ref_num_flash': np.zeros(num_data_records, dtype=np.uint16),
        }
        sig_pix = np.zeros((num_data_records, config.sig_num_wls), dtype=np.uint16)
        ref_pix = np.zeros((num_data_records, config.ref_num_wls), dtype=np.uint16)

        # Parse data records
        for i in range(num_data_records):
            records['record_id'][i] = struct.unpack('<H', f.read(2))[0]

            # Read datetime (6 bytes: year, month, day, hour, minute, second)
            dt_bytes = struct.unpack('<6B', f.read(6))
            try:
                records['date'][i] = datetime(
                    1900 + dt_bytes[0], dt_bytes[1], dt_bytes[2],
                    dt_bytes[3], dt_bytes[4], dt_bytes[5]
                )
            except ValueError:
                records['date'][i] = None

            records['input_voltage'][i] = struct.unpack('<H', f.read(2))[0] / 100.0
            records['water_temp'][i] = struct.unpack('<H', f.read(2))[0] / 100.0 - 10.0
            records['depth'][i] = struct.unpack('<H', f.read(2))[0] / 100.0 - 10.0
            records['board_temp'][i] = struct.unpack('<H', f.read(2))[0] / 100.0 - 10.0
            records['board_humid'][i] = struct.unpack('<B', f.read(1))[0]
            records['lamp_freq'][i] = struct.unpack('<B', f.read(1))[0]
            records['sig_avg'][i] = struct.unpack('<B', f.read(1))[0]
            records['ref_avg'][i] = struct.unpack('<B', f.read(1))[0]
            records['sig_exp'][i] = struct.unpack('<I', f.read(4))[0]
            records['ref_exp'][i] = struct.unpack('<I', f.read(4))[0]
            records['sig_num_flash'][i] = struct.unpack('<H', f.read(2))[0]
            records['ref_num_flash'][i] = struct.unpack('<H', f.read(2))[0]

            sig_pix[i, :] = np.array(
                struct.unpack(f'<{config.sig_num_wls}H', f.read(2 * config.sig_num_wls))
            )
            ref_pix[i, :] = np.array(
                struct.unpack(f'<{config.ref_num_wls}H', f.read(2 * config.ref_num_wls))
            )

    # Create DataFrame
    df = pd.DataFrame(records)
    df['sig_pix'] = list(sig_pix)
    df['ref_pix'] = list(ref_pix)

    return HyperaData(config=config, data=df)


def load_calibration(filepath: str) -> HyperaCalibration:
    """
    Load a Hyper-a calibration file (.mat format).

    Parameters
    ----------
    filepath : str
        Path to the calibration .mat file

    Returns
    -------
    HyperaCalibration
        Calibration data object
    """
    mat = sio.loadmat(filepath, squeeze_me=True)
    cal_struct = mat['cal']

    # Handle structured array from MATLAB
    if cal_struct.dtype.names is not None:
        wl = np.atleast_1d(cal_struct['wl'].item()).flatten()
        r = float(np.atleast_1d(cal_struct['r'].item()).flatten()[0])
        r_0 = float(np.atleast_1d(cal_struct['r_0'].item()).flatten()[0])
        rho = np.atleast_1d(cal_struct['rho'].item()).flatten()

        date = ''
        if 'date' in cal_struct.dtype.names:
            date_val = cal_struct['date'].item()
            if isinstance(date_val, np.ndarray):
                date = str(date_val.flatten()[0]) if date_val.size > 0 else ''
            else:
                date = str(date_val)

        proc_ver = 1.0
        if 'procVer' in cal_struct.dtype.names:
            proc_ver = float(np.atleast_1d(cal_struct['procVer'].item()).flatten()[0])

        serial_number = 0
        if 'SN' in cal_struct.dtype.names:
            serial_number = int(np.atleast_1d(cal_struct['SN'].item()).flatten()[0])

        spot_absorp = None
        if 'spotAbsorp' in cal_struct.dtype.names:
            spot_absorp = np.atleast_1d(cal_struct['spotAbsorp'].item()).flatten()
    else:
        raise ValueError("Unexpected calibration file format")

    return HyperaCalibration(
        wl=wl,
        r=r,
        r_0=r_0,
        rho=rho,
        date=date,
        proc_ver=proc_ver,
        serial_number=serial_number,
        spot_absorp=spot_absorp
    )


def load_mat_data(filepath: str) -> HyperaData:
    """
    Load Hyper-a data from a .mat file.

    Parameters
    ----------
    filepath : str
        Path to the .mat file containing config and dataTable

    Returns
    -------
    HyperaData
        Data container with config and DataFrame

    Notes
    -----
    The .mat file should contain 'config' and 'dataTable' variables.
    Due to MATLAB table format limitations, the dataTable must be saved
    as a struct array in MATLAB for proper loading in Python.
    """
    mat = sio.loadmat(filepath, squeeze_me=True, simplify_cells=True)

    # Load config (may be a list if multiple configs)
    config_data = mat['config']
    if isinstance(config_data, list):
        config_dict = config_data[0]
    elif isinstance(config_data, np.ndarray) and config_data.dtype.names:
        config_dict = {name: config_data[name].item() for name in config_data.dtype.names}
    else:
        config_dict = config_data

    config = HyperaConfig(
        serial_number=int(config_dict.get('serialNumber', 0)),
        firmware_ver=float(config_dict.get('firmwareVer', 0)),
        sig_spec_sn=int(config_dict.get('sigSpecSN', 0)),
        sef_spec_sn=int(config_dict.get('sefSpecSN', 0)),
        sig_num_wls=int(config_dict.get('sigNumWls', 0)),
        ref_num_wls=int(config_dict.get('refNumWls', 0)),
        config_byte=int(config_dict.get('configByte', 0)),
        main_board_rev=str(config_dict.get('mainBoardRev', '')),
        pump_flush_sec=int(config_dict.get('pumpFlushSec', 0)),
        sequence_interval_sec=int(config_dict.get('sequenceInterval_sec', 0)),
        burst_interval_min=int(config_dict.get('burstInterval_min', 0)),
        sequences_per_burst=int(config_dict.get('sequencesPerBurst', 0)),
        sig_wls=np.atleast_1d(config_dict.get('sigWls', np.array([]))),
        ref_wls=np.atleast_1d(config_dict.get('refWls', np.array([]))),
        meas_id=np.atleast_1d(config_dict.get('measID', np.zeros(8))),
    )

    # Parse config byte flags
    config.auto_exposure = bool(config.config_byte & 1)
    config.sig_lin_corr = bool(config.config_byte & 2)
    config.ref_lin_corr = bool(config.config_byte & 4)
    config.pump_enabled = bool(config.config_byte & 8)
    config.auto_start = bool(config.config_byte & 16)
    config.switch_start = bool(config.config_byte & 32)

    # Load linearity coefficients if present
    if 'sigSpecLinCoeff' in config_dict:
        config.sig_spec_lin_coeff = np.atleast_1d(config_dict['sigSpecLinCoeff'])
    if 'refSpecLinCoeff' in config_dict:
        config.ref_spec_lin_coeff = np.atleast_1d(config_dict['refSpecLinCoeff'])

    # Note: MATLAB table objects cannot be directly loaded by scipy.io
    # The dataTable should be converted to a struct array in MATLAB first
    # or the data should be loaded from .bin files
    data_table = mat.get('dataTable')

    # Check if it's a MATLAB opaque object (table)
    if data_table is not None and hasattr(data_table, 'dtype'):
        if 'MatlabOpaque' in str(type(data_table.flat[0])) if data_table.size > 0 else False:
            raise ValueError(
                "MATLAB table objects cannot be directly loaded. "
                "Please save the dataTable as a struct array in MATLAB using: "
                "table2struct(dataTable, 'ToScalar', false)"
            )

    # Create empty DataFrame if dataTable couldn't be loaded
    df = pd.DataFrame()

    return HyperaData(config=config, data=df)


def import_hypera_data(path_or_data: Union[str, HyperaData, Dict]) -> HyperaData:
    """
    Import Hyper-a data from file or use provided data structure.

    Parameters
    ----------
    path_or_data : str, HyperaData, or dict
        Either a file path (.bin or .mat) or an existing HyperaData object

    Returns
    -------
    HyperaData
        Data container with config and DataFrame
    """
    if isinstance(path_or_data, HyperaData):
        return path_or_data

    if isinstance(path_or_data, dict):
        # Assume it's a dict with config and data
        if 'config' in path_or_data and 'data' in path_or_data:
            return HyperaData(
                config=path_or_data['config'],
                data=path_or_data['data']
            )
        raise ValueError("Dict must contain 'config' and 'data' keys")

    if isinstance(path_or_data, str):
        if path_or_data.endswith('.bin'):
            return read_bin(path_or_data)
        elif path_or_data.endswith('.mat'):
            return load_mat_data(path_or_data)
        else:
            raise ValueError("File must be .bin or .mat")

    raise ValueError(f"Unsupported input type: {type(path_or_data)}")
