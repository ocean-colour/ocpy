""" I/O for Remote Sensing data and more """
import os
import numpy as np


def load_loisel_2023_pca(pca_file:str='pca_ab_33_Rrs.npz',
                         back_scatt:str='bb'):

    # Load up data
    l23_path = os.path.join(os.getenv('OS_COLOR'),
                            'data', 'Loisel2023')
    outfile = os.path.join(l23_path, pca_file)

    d = np.load(outfile)
    nparam = d['a'].shape[1]+d[back_scatt].shape[1]
    ab = np.zeros((d['a'].shape[0], nparam))
    ab[:,0:d['a'].shape[1]] = d['a']
    ab[:,d['a'].shape[1]:] = d[back_scatt]

    Rs = d['Rs']

    # Return
    return ab, Rs, d