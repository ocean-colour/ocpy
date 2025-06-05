import numpy as np
from ocpy.utils import coords

import pytest


def test_dms_to_decimal():
    # Example usage
    # For example, 40° 26' 46" N, 79° 58' 56" W (Pittsburgh, PA)
    lat_deg, lat_min, lat_sec, lat_dir = 40, 26, 46, 'N'
    lon_deg, lon_min, lon_sec, lon_dir = 79, 58, 56, 'W'

    lat_decimal = coords.dms_to_decimal(lat_deg, lat_min, lat_sec, lat_dir)
    lon_decimal = coords.dms_to_decimal(lon_deg, lon_min, lon_sec, lon_dir)

    # Test
    assert np.isclose(lat_decimal, 40.446111, atol=1e-6)
    assert np.isclose(lon_decimal, -79.982222, atol=1e-6)

def test_dmsstr_to_decimal():

    # Example usage
    lat_str1 = "40° 26' 46\" N"
    lon_str1 = "79° 58' 56\" W"
    lat_str2 = "40d26m46sN"
    lon_str2 = "79d58m56sW"

    for lat_str, lon_str in [(lat_str1, lon_str1), (lat_str2, lon_str2)]:
        # Parse the DMS strings
        lat_decimal = coords.parse_dms_string(lat_str)
        lon_decimal = coords.parse_dms_string(lon_str)

        # Test
        assert np.isclose(lat_decimal, 40.446111, atol=1e-6)
        assert np.isclose(lon_decimal, -79.982222, atol=1e-6)