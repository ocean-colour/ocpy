""" Tests for phytoplankton """""
import os

from oceancolor.polarize import load_data

import pytest

from IPython import embed

def test_load_koetner2021():
    # Lagoon
    df, psis, vsf = load_data.koetner2021(sheet='Lagoon')

    assert vsf.shape[0] == len(psis)
    assert vsf.shape[1] == len(df)

    # BS
    df, psis, vsf = load_data.koetner2021(sheet='BS')