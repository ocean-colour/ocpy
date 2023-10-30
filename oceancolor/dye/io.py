""" I/O for Dye """

import os

import numpy as np

import h5py

from IPython import embed


def extract_cube(mat_file:str, cuts:list=[]):
    f = h5py.File(mat_file, 'r')

    # Intensities
    allI = f['hyperspec']['I'][:]

    # Wavelengths
    wave = f['hyperspec']['wavelengths'][:].flatten()

    # Coords
    lat = f['hyperspec']['lat'][:].flatten()
    lon = f['hyperspec']['lon'][:].flatten()

    # Return
    return wave, lat, lon, allI


if __name__ == '__main__':

    # Process full file
    mat_file = os.path.join(os.getenv('OS_COLOR'), 'data', 'Dye', 
                            '20230120093344to20230120093402PST.mat')
    outfile = os.path.join(os.getenv('OS_COLOR'), 'data', 'Dye',
                           '20230120093344to20230120093402PST.npz')    

    process_matlab(mat_file, outfile)