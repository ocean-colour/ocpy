""" I/O functions for ls2 """

import os
from importlib.resources import files
import numpy as np

import pandas


def load_LUT():
    """ Load the LUT from the package data """
    filename = files('ocpy').joinpath(
        os.path.join('data', 'LS2', 'LS2_LUT.npz'))
    return np.load(filename)

def load_Kd_tables():
    weights_1_file = files('ocpy').joinpath(
        os.path.join('data', 'LS2', 'weights_1.csv'))
    weights_2_file = files('ocpy').joinpath(
        os.path.join('data', 'LS2', 'weights_2.csv'))
    train_file = files('ocpy').joinpath(
        os.path.join('data', 'LS2', 'train_switch.csv'))
    # Load
    weights_1 = pandas.read_csv(weights_1_file)
    weights_2 = pandas.read_csv(weights_2_file)
    train_switch = pandas.read_csv(train_file)

    # Return
    return weights_1, weights_2, train_switch