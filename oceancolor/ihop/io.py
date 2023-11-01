""" I/O for Remote Sensing data and more """
import os
import numpy as np

import torch

from importlib import resources

#from oceancolor.ihop.nn import SimpleNet


def load_loisel_2023_pca(back_scatt:str='bb'):

    # Load up data
    l23_path = os.path.join(resources.files('oceancolor'),
                            'data', 'PCA')
    l23_a_file = os.path.join(l23_path, 'pca_L23_X4Y0_a_N3.npz')
    l23_bb_file = os.path.join(l23_path, 'pca_L23_X4Y0_bb_N3.npz')

    d_a = np.load(l23_a_file)
    d_bb = np.load(l23_bb_file)
    nparam = d_a['Y'].shape[1]+d_bb['Y'].shape[1]
    ab = np.zeros((d_a['Y'].shape[0], nparam))
    ab[:,0:d_a['Y'].shape[1]] = d_a['Y']
    ab[:,d_a['Y'].shape[1]:] = d_bb['Y']

    Rs = d_a['Rs']

    # Return
    return ab, Rs, d_a, d_bb

def load_nn(root:str):
    model_file = os.path.join(
        resources.files('oceancolor'), 
        'data', 'ihop', f'{root}.pth')
    model = torch.load(model_file)

    return model