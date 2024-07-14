""" Tests for phytoplankton """""
import os

from oceancolor.ph import io

import pytest

from IPython import embed

def test_load_tables():
    df = io.load_tables()

#pytest.set_trace()