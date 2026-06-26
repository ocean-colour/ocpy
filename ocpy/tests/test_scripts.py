"""Tests for the scripts module"""

import argparse
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
