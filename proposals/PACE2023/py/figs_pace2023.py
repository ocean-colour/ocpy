""" Figures for the PACE 2023 proposal """

# imports
from importlib import reload
import os
import xarray

import numpy as np
from scipy import stats
from scipy.interpolate import interp1d 

from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator 

from oceancolor.ihop import pca as ihop_pca
from oceancolor.utils import plotting 

mpl.rcParams['font.family'] = 'stixgeneral'


import seaborn as sns

import pandas


from IPython import embed


def gen_cb(img, lbl, csz = 17.):
    cbaxes = plt.colorbar(img, pad=0., fraction=0.030)
    cbaxes.set_label(lbl, fontsize=csz)
    cbaxes.ax.tick_params(labelsize=csz)


def fig_l23_tara_pca(outfile='fig_l23_tara_pca.png'):

    # Load up
    L23_Tara_pca_N20 = ihop_pca.load_pca('pca_L23_X4Y0_Tara_a_N20.npz')
    L23_Tara_pca_N3 = ihop_pca.load_pca('pca_L23_X4Y0_Tara_a_N3.npz')


    fig = plt.figure(figsize=(12,6))
    plt.clf()
    gs = gridspec.GridSpec(1,2)

    # #####################################################
    # PDF
    ax_var = plt.subplot(gs[0])

    ax_var.plot(np.arange(L23_Tara_pca_N20['explained_variance'].size)+1,
        np.cumsum(L23_Tara_pca_N20['explained_variance']), 'o-')
    ax_var.set_xlim(0., 10)
    ax_var.set_xlabel('Number of Components')
    ax_var.set_ylabel('Cumulative Explained Variance')

    # Finish
    for ax in [ax_var]:
        plotting.set_fontsize(ax, 15)


    
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)


    # PCA
    if flg & (2**0):
        fig_l23_tara_pca()


# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- Joint PDFs of all 4 lines
        #flg += 2 ** 1  # 2 -- SO CDF
        #flg += 2 ** 2  # 3 -- DOY vs Offshore, 1 by 1
    else:
        flg = sys.argv[1]

    main(flg)