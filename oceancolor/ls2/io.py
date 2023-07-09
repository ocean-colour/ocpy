""" I/O functions for ls2 """

import os
from pkg_resources import resource_filename
import numpy as np


def load_LUT():
    """ Load the LUT from the package data """
    filename = resource_filename('oceancolor', 
                                 os.path.join('data', 'LS2', 'LS2_LUT.npz'))
    return np.load(filename)