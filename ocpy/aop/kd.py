""" Methods related to Kd """

import numpy as np


def calc_kd_lee(a:np.ndarray, bb:np.ndarray, theta_sun:float=0.):


    Kd = (1. + 1.005*theta_sun) * (a + bb) / np.cos(np.radians(theta_sun))