"""Tests for the plot_oc_scene script"""

import argparse
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from ocpy.scripts import plot_oc_scene


class TestParser:
    """Tests for the plot_oc_scene argument parser"""

    def test_parser_returns_argumentparser(self):
        """parser() should return an ArgumentParser instance"""
        p = plot_oc_scene.parser()
        assert isinstance(p, argparse.ArgumentParser)

    def test_parser_with_default_channels(self):
        """Parser should use default channels when not specified"""
        p = plot_oc_scene.parser()
        args = p.parse_args(['granule.nc'])
        assert args.granule == 'granule.nc'
        assert args.channels == "350,400,452,552,600,650,700,750,801"
        assert args.ncols == 3

    def test_parser_with_custom_channels(self):
        """Parser should accept custom channel specification"""
        p = plot_oc_scene.parser()
        args = p.parse_args(['granule.nc', '--channels', '400,450,550,650'])
        assert args.granule == 'granule.nc'
        assert args.channels == '400,450,550,650'

    def test_parser_with_output(self):
        """Parser should accept output file argument"""
        p = plot_oc_scene.parser()
        args = p.parse_args(['granule.nc', '-o', 'output.html'])
        assert args.output == 'output.html'

    def test_parser_with_ncols(self):
        """Parser should accept ncols argument"""
        p = plot_oc_scene.parser()
        args = p.parse_args(['granule.nc', '--ncols', '4'])
        assert args.ncols == 4

    def test_parser_help_does_not_crash(self):
        """Parser -h should not crash"""
        p = plot_oc_scene.parser()
        with pytest.raises(SystemExit) as exc_info:
            p.parse_args(['-h'])
        assert exc_info.value.code == 0


class TestDetectFileType:
    """Tests for detect_file_type function"""

    def test_detect_pace_l2_from_title(self):
        """Should detect PACE L2 from title attribute"""
        mock_ds = MagicMock()
        mock_ds.ncattrs.return_value = ['title']
        mock_ds.getncattr.return_value = 'PACE OCI L2 Ocean Color'
        mock_ds.groups = {'geophysical_data': Mock(), 'navigation_data': Mock()}

        with patch('ocpy.scripts.plot_oc_scene.Dataset', return_value=mock_ds):
            result = plot_oc_scene.detect_file_type('test.nc')
            assert result == 'PACE_L2'

    def test_detect_pace_l1c_from_title(self):
        """Should detect PACE L1C from title attribute"""
        mock_ds = MagicMock()
        mock_ds.ncattrs.return_value = ['title']
        mock_ds.getncattr.return_value = 'PACE OCI L1C'
        mock_ds.groups = {'observation_data': Mock()}

        with patch('ocpy.scripts.plot_oc_scene.Dataset', return_value=mock_ds):
            result = plot_oc_scene.detect_file_type('test.nc')
            assert result == 'PACE_L1C'

    def test_detect_modis_from_instrument(self):
        """Should detect MODIS from instrument attribute"""
        mock_ds = MagicMock()
        mock_ds.ncattrs.return_value = ['instrument']
        mock_ds.getncattr.return_value = 'MODIS-Aqua'
        mock_ds.groups = {}

        with patch('ocpy.scripts.plot_oc_scene.Dataset', return_value=mock_ds):
            result = plot_oc_scene.detect_file_type('test.nc')
            assert result == 'MODIS_L2'

    def test_detect_unknown_format(self):
        """Should return UNKNOWN for unrecognized files"""
        mock_ds = MagicMock()
        mock_ds.ncattrs.return_value = []
        mock_ds.groups = {}

        with patch('ocpy.scripts.plot_oc_scene.Dataset', return_value=mock_ds):
            result = plot_oc_scene.detect_file_type('test.nc')
            assert result == 'UNKNOWN'


class TestLoadPACEL2Scene:
    """Tests for load_pace_l2_scene function"""

    def test_load_pace_l2_scene_structure(self):
        """Should return correct data structure"""
        # Create mock data
        nx, ny, nwl = 100, 50, 10
        mock_rrs = np.random.rand(nx, ny, nwl)
        mock_lats = np.random.rand(nx, ny) * 180 - 90
        mock_lons = np.random.rand(nx, ny) * 360 - 180
        mock_wls = np.array([400, 410, 420, 430, 440, 450, 460, 470, 480, 490])
        mock_flags = np.zeros((nx, ny), dtype=np.int32)

        # Mock Dataset context manager
        mock_ds = MagicMock()
        mock_ds.__enter__ = Mock(return_value=mock_ds)
        mock_ds.__exit__ = Mock(return_value=None)

        mock_gd = MagicMock()
        mock_nav = MagicMock()
        mock_sbp = MagicMock()

        mock_ds.groups = {
            'geophysical_data': mock_gd,
            'navigation_data': mock_nav,
            'sensor_band_parameters': mock_sbp
        }

        mock_nav.variables = {
            'latitude': Mock(__getitem__=lambda s, k: mock_lats),
            'longitude': Mock(__getitem__=lambda s, k: mock_lons)
        }
        mock_sbp.__getitem__ = lambda s, k: mock_wls
        mock_gd.variables = {
            'Rrs': Mock(__getitem__=lambda s, k: mock_rrs),
            'l2_flags': Mock(__getitem__=lambda s, k: mock_flags)
        }

        with patch('ocpy.scripts.plot_oc_scene.Dataset', return_value=mock_ds):
            wavelengths = [400, 450, 490]
            data, lats, lons, flags, actual_wls = plot_oc_scene.load_pace_l2_scene(
                'test.nc', wavelengths
            )

            assert data.shape == (len(wavelengths), nx, ny)
            assert lats.shape == (nx, ny)
            assert lons.shape == (nx, ny)
            assert flags.shape == (nx, ny)
            assert len(actual_wls) == len(wavelengths)


class TestLoadPACEL1CScene:
    """Tests for load_pace_l1c_scene function"""

    def test_load_pace_l1c_scene_structure(self):
        """Should return correct data structure"""
        nx, ny, nwl = 100, 50, 10
        mock_rhot = np.random.rand(nx, ny, nwl)
        mock_lats = np.random.rand(nx, ny) * 180 - 90
        mock_lons = np.random.rand(nx, ny) * 360 - 180
        mock_wls = np.array([400, 410, 420, 430, 440, 450, 460, 470, 480, 490])

        mock_ds = MagicMock()
        mock_ds.__enter__ = Mock(return_value=mock_ds)
        mock_ds.__exit__ = Mock(return_value=None)

        mock_obs = MagicMock()
        mock_nav = MagicMock()
        mock_sbp = MagicMock()

        mock_ds.groups = {
            'observation_data': mock_obs,
            'navigation_data': mock_nav,
            'sensor_band_parameters': mock_sbp
        }

        mock_nav.variables = {
            'latitude': Mock(__getitem__=lambda s, k: mock_lats),
            'longitude': Mock(__getitem__=lambda s, k: mock_lons)
        }
        mock_sbp.__getitem__ = lambda s, k: mock_wls
        mock_obs.variables = {
            'rhot': Mock(__getitem__=lambda s, k: mock_rhot)
        }

        with patch('ocpy.scripts.plot_oc_scene.Dataset', return_value=mock_ds):
            wavelengths = [400, 450, 490]
            data, lats, lons, flags, actual_wls = plot_oc_scene.load_pace_l1c_scene(
                'test.nc', wavelengths
            )

            assert data.shape == (len(wavelengths), nx, ny)
            assert lats.shape == (nx, ny)
            assert lons.shape == (nx, ny)
            assert flags.shape == (nx, ny)
            assert len(actual_wls) == len(wavelengths)


class TestLoadMODISL2Scene:
    """Tests for load_modis_l2_scene function"""

    def test_wavelength_matching(self):
        """Should match requested wavelengths to MODIS bands"""
        nx, ny = 100, 50
        mock_lats = np.random.rand(nx, ny) * 180 - 90
        mock_lons = np.random.rand(nx, ny) * 360 - 180
        mock_flags = np.zeros((nx, ny), dtype=np.int32)

        mock_ds = MagicMock()
        mock_ds.__enter__ = Mock(return_value=mock_ds)
        mock_ds.__exit__ = Mock(return_value=None)

        mock_gd = MagicMock()
        mock_nav = MagicMock()

        mock_ds.groups = {
            'geophysical_data': mock_gd,
            'navigation_data': mock_nav
        }

        mock_nav.variables = {
            'latitude': Mock(__getitem__=lambda s, k: mock_lats),
            'longitude': Mock(__getitem__=lambda s, k: mock_lons)
        }

        # Create mock variables for MODIS bands
        def mock_getitem(key):
            if key == 'l2_flags':
                return Mock(__getitem__=lambda s, k: mock_flags)
            elif key.startswith('Rrs_'):
                return Mock(__getitem__=lambda s, k: np.random.rand(nx, ny))
            raise KeyError(key)

        mock_gd.variables = MagicMock()
        mock_gd.variables.__contains__ = lambda s, k: k in [
            'Rrs_412', 'Rrs_443', 'Rrs_488', 'l2_flags'
        ]
        mock_gd.variables.__getitem__ = mock_getitem

        with patch('ocpy.scripts.plot_oc_scene.Dataset', return_value=mock_ds):
            # Request wavelengths close to MODIS bands
            wavelengths = [410, 445, 490]  # Should match to 412, 443, 488
            data, lats, lons, flags, actual_wls = plot_oc_scene.load_modis_l2_scene(
                'test.nc', wavelengths
            )

            # Check that wavelengths were matched correctly
            assert 412 in actual_wls  # 410 matched to 412
            assert 443 in actual_wls  # 445 matched to 443
            assert 488 in actual_wls  # 490 matched to 488


class TestCreateScenePlot:
    """Tests for create_scene_plot function"""

    def test_handles_masked_data(self):
        """Should properly handle masked/invalid data"""
        # Create test data with some masked values
        n_wavelengths = 3
        nx, ny = 50, 40
        data = np.random.rand(n_wavelengths, nx, ny)
        lats = np.random.rand(nx, ny) * 180 - 90
        lons = np.random.rand(nx, ny) * 360 - 180
        flags = np.zeros((nx, ny), dtype=np.int32)

        # Mark some pixels as invalid
        flags[10:20, 15:25] = 1
        data[0, 5:10, 5:10] = np.nan

        wavelengths = np.array([400, 500, 600])

        # Mock bokeh show function to prevent browser launch
        with patch('ocpy.scripts.plot_oc_scene.show') as mock_show, \
             patch('ocpy.scripts.plot_oc_scene.output_file'):

            # Should not raise any errors
            plot_oc_scene.create_scene_plot(
                data, lats, lons, flags, wavelengths,
                filename='test.nc'
            )

            # Verify show was called
            mock_show.assert_called_once()

    def test_grid_layout_dimensions(self):
        """Should create correct grid dimensions"""
        n_wavelengths = 7  # Should create 3x3 grid with 2 empty spots
        nx, ny = 50, 40
        data = np.random.rand(n_wavelengths, nx, ny)
        lats = np.random.rand(nx, ny) * 180 - 90
        lons = np.random.rand(nx, ny) * 360 - 180
        flags = np.zeros((nx, ny), dtype=np.int32)
        wavelengths = np.arange(400, 400 + n_wavelengths * 50, 50)

        with patch('ocpy.scripts.plot_oc_scene.show') as mock_show, \
             patch('ocpy.scripts.plot_oc_scene.output_file'), \
             patch('ocpy.scripts.plot_oc_scene.gridplot') as mock_gridplot:

            plot_oc_scene.create_scene_plot(
                data, lats, lons, flags, wavelengths,
                filename='test.nc',
                ncols=3
            )

            # Check that gridplot was called
            mock_gridplot.assert_called_once()

            # Get the grid argument (first positional argument)
            grid = mock_gridplot.call_args[0][0]

            # Should have 3 rows (7 plots + 2 None = 9 slots / 3 cols)
            assert len(grid) == 3


class TestMainFunction:
    """Tests for main entry point"""

    def test_invalid_wavelength_format(self, capsys):
        """Should exit with error for invalid wavelength format"""
        args = argparse.Namespace(
            granule='test.nc',
            channels='400,invalid,600',
            ncols=3,
            output=None
        )

        with pytest.raises(SystemExit) as exc_info:
            plot_oc_scene.main(args)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert 'Invalid wavelength format' in captured.out

    def test_file_not_found(self, capsys):
        """Should exit with error for missing file"""
        args = argparse.Namespace(
            granule='nonexistent.nc',
            channels='400,500,600',
            ncols=3,
            output=None
        )

        with pytest.raises(SystemExit) as exc_info:
            plot_oc_scene.main(args)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert 'File not found' in captured.out

    def test_unsupported_file_type(self, capsys):
        """Should exit with error for unsupported file type"""
        args = argparse.Namespace(
            granule='test.nc',
            channels='400,500,600',
            ncols=3,
            output=None
        )

        with patch('os.path.exists', return_value=True), \
             patch('ocpy.scripts.plot_oc_scene.detect_file_type', return_value='UNKNOWN'):

            with pytest.raises(SystemExit) as exc_info:
                plot_oc_scene.main(args)

            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert 'Unsupported file type' in captured.out
