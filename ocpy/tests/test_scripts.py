"""Tests for the scripts module"""

import argparse
import numpy as np
import xarray as xr
import pytest

from ocpy.scripts import plot_rrs


class TestPlotRrsParser:
    """Tests for the plot_rrs argument parser"""

    def test_parser_returns_argumentparser(self):
        """parser() should return an ArgumentParser instance"""
        p = plot_rrs.parser()
        assert isinstance(p, argparse.ArgumentParser)

    def test_parser_with_latlon(self):
        """Parser should accept lat/lon arguments"""
        p = plot_rrs.parser()
        args = p.parse_args(['granule.nc', '--lat', '34.5', '--lon', '-120.3'])
        assert args.granule == 'granule.nc'
        assert args.lat == 34.5
        assert args.lon == -120.3
        assert args.x is None
        assert args.y is None

    def test_parser_with_xy(self):
        """Parser should accept x/y arguments"""
        p = plot_rrs.parser()
        args = p.parse_args(['granule.nc', '--x', '500', '--y', '800'])
        assert args.granule == 'granule.nc'
        assert args.x == 500
        assert args.y == 800
        assert args.lat is None
        assert args.lon is None

    def test_parser_with_output(self):
        """Parser should accept output file argument"""
        p = plot_rrs.parser()
        args = p.parse_args(['granule.nc', '--lat', '34.5', '--lon', '-120.3',
                             '-o', 'output.html'])
        assert args.output == 'output.html'

    def test_parser_help_does_not_crash(self):
        """Parser -h should not crash"""
        p = plot_rrs.parser()
        with pytest.raises(SystemExit) as exc_info:
            p.parse_args(['-h'])
        assert exc_info.value.code == 0


class TestFindNearestPixel:
    """Tests for find_nearest_pixel function"""

    def test_find_exact_match(self):
        """Should find exact pixel when lat/lon matches"""
        # Create a simple test dataset
        lat = np.array([[10.0, 10.0], [20.0, 20.0]])
        lon = np.array([[-120.0, -110.0], [-120.0, -110.0]])
        xds = xr.Dataset(
            coords={
                'latitude': (['x', 'y'], lat),
                'longitude': (['x', 'y'], lon),
            }
        )

        x_idx, y_idx = plot_rrs.find_nearest_pixel(xds, 20.0, -110.0)
        assert x_idx == 1
        assert y_idx == 1

    def test_find_nearest(self):
        """Should find nearest pixel when no exact match"""
        lat = np.array([[10.0, 10.0], [20.0, 20.0]])
        lon = np.array([[-120.0, -110.0], [-120.0, -110.0]])
        xds = xr.Dataset(
            coords={
                'latitude': (['x', 'y'], lat),
                'longitude': (['x', 'y'], lon),
            }
        )

        # Request a point closer to (20, -110)
        x_idx, y_idx = plot_rrs.find_nearest_pixel(xds, 19.0, -111.0)
        assert x_idx == 1
        assert y_idx == 1

    def test_handles_nan_in_coords(self):
        """Should handle NaN values in coordinates"""
        lat = np.array([[np.nan, 10.0], [20.0, 20.0]])
        lon = np.array([[np.nan, -110.0], [-120.0, -110.0]])
        xds = xr.Dataset(
            coords={
                'latitude': (['x', 'y'], lat),
                'longitude': (['x', 'y'], lon),
            }
        )

        # Should find valid pixel, not NaN
        x_idx, y_idx = plot_rrs.find_nearest_pixel(xds, 10.0, -110.0)
        assert x_idx == 0
        assert y_idx == 1


class TestExtractSpectrum:
    """Tests for extract_spectrum function"""

    def test_extract_valid_spectrum(self):
        """Should extract Rrs and uncertainty at given pixel"""
        # Create test dataset
        wavelength = np.array([400, 450, 500, 550, 600])
        rrs_data = np.zeros((3, 3, 5))
        rrs_data[1, 1, :] = [0.001, 0.002, 0.003, 0.002, 0.001]
        rrs_unc_data = np.zeros((3, 3, 5))
        rrs_unc_data[1, 1, :] = [0.0001, 0.0002, 0.0003, 0.0002, 0.0001]

        xds = xr.Dataset(
            data_vars={
                'Rrs': (['x', 'y', 'wl'], rrs_data),
                'Rrs_unc': (['x', 'y', 'wl'], rrs_unc_data),
            },
            coords={
                'wavelength': ('wl', wavelength),
            }
        )
        flags = np.zeros((3, 3), dtype=int)

        wl, rrs, rrs_unc = plot_rrs.extract_spectrum(xds, 1, 1, flags)

        np.testing.assert_array_equal(wl, wavelength)
        np.testing.assert_array_almost_equal(rrs, [0.001, 0.002, 0.003, 0.002, 0.001])
        np.testing.assert_array_almost_equal(rrs_unc, [0.0001, 0.0002, 0.0003, 0.0002, 0.0001])

    def test_warns_on_nonzero_flag(self, capsys):
        """Should print warning when quality flag is non-zero"""
        wavelength = np.array([400, 450, 500])
        rrs_data = np.ones((2, 2, 3)) * 0.001
        rrs_unc_data = np.ones((2, 2, 3)) * 0.0001

        xds = xr.Dataset(
            data_vars={
                'Rrs': (['x', 'y', 'wl'], rrs_data),
                'Rrs_unc': (['x', 'y', 'wl'], rrs_unc_data),
            },
            coords={
                'wavelength': ('wl', wavelength),
            }
        )
        flags = np.array([[0, 1], [0, 0]], dtype=int)

        plot_rrs.extract_spectrum(xds, 0, 1, flags)
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "non-zero quality flag" in captured.out

    def test_exits_on_all_invalid_data(self):
        """Should exit when all Rrs data is invalid"""
        wavelength = np.array([400, 450, 500])
        rrs_data = np.full((2, 2, 3), np.nan)
        rrs_unc_data = np.full((2, 2, 3), np.nan)

        xds = xr.Dataset(
            data_vars={
                'Rrs': (['x', 'y', 'wl'], rrs_data),
                'Rrs_unc': (['x', 'y', 'wl'], rrs_unc_data),
            },
            coords={
                'wavelength': ('wl', wavelength),
            }
        )
        flags = np.zeros((2, 2), dtype=int)

        with pytest.raises(SystemExit) as exc_info:
            plot_rrs.extract_spectrum(xds, 0, 0, flags)
        assert exc_info.value.code == 1
