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

from pypeit.core import fitting
from pypeit.bspline import bspline

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
    N=3
    L23_Tara_pca = ihop_pca.load_pca(f'pca_L23_X4Y0_Tara_a_N{N}.npz')
    wave = L23_Tara_pca['wavelength']


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

    # #####################################################
    # Reconstructions
    ax_recon = plt.subplot(gs[1])

    idx = 1000  # L23 
    orig, recon = ihop_pca.reconstruct(L23_Tara_pca, idx)
    lbl = 'L23'
    ax_recon.plot(wave, orig,  label=lbl)
    ax_recon.plot(wave, recon, 'r:', label=f'L23 Model (N={N})')

    idx = 100000  # L23 
    orig, recon = ihop_pca.reconstruct(L23_Tara_pca, idx)
    lbl = 'Tara'
    ax_recon.plot(wave, orig,  'g', label=lbl)
    ax_recon.plot(wave, recon, color='orange', ls=':', label=f'Tara Model (N={N})')
    
    
    #
    ax_recon.set_xlabel('Wavelength (nm)')
    ax_recon.set_ylabel(r'$a(\lambda)$')
    ax_recon.set_ylim(0., 1.1*np.max(recon))
    ax_recon.legend(fontsize=15.)
    
    # Stats
    #rms = np.sqrt(np.mean((var.a.data[idx] - a_recon3[idx])**2))
    #max_dev = np.max(np.abs((var.a.data[idx] - a_recon3[idx])/a_recon3[idx]))
    #ax.text(0.05, 0.7, f'RMS={rms:0.4f}, max '+r'$\delta$'+f'={max_dev:0.2f}',
    #        transform=ax.transAxes,
    #          fontsize=16., ha='left', color='k')  


    # Finish
    for ax in [ax_var, ax_recon]:
        plotting.set_fontsize(ax, 15)


    
    plt.tight_layout()#pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")


def fig_bspline_tara(outfile='fig_bspline_tara.png'):

    # Load up
    wv_grid, tara_a_water, l23_a = ihop_pca.load_tara()
    wv_grid = wv_grid.astype(np.float64)

    L23_Tara_pca = ihop_pca.load_pca(f'pca_L23_X4Y0_Tara_a_N3.npz')

    imx = np.argmax(tara_a_water[:,3])


    fig = plt.figure(figsize=(12,6))
    plt.clf()
    gs = gridspec.GridSpec(1,2)

    # #####################################################
    # RMS
    ax_rms= plt.subplot(gs[0])

    # #####################################################
    # Spectra
    ax_spec= plt.subplot(gs[1])

    # B-spline it
    ifit = 1
    everyns = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    for clr, idx in zip(['b', 'g'], [100000, imx]):
        orig, recon = ihop_pca.reconstruct(L23_Tara_pca, idx)
        orig = orig.astype(np.float64)
        sv_fits = [] 
        rmss = []
        for everyn in everyns:
            my_bspline = bspline(wv_grid, nord=3,
                        everyn=everyn)
            code, yfit = my_bspline.fit(wv_grid, orig, np.ones_like(orig))
            sv_fits += [yfit]
            # RMS
            rmss += [np.sqrt(np.mean((orig - yfit)**2))]

        # Plot RMS vs. everyn
        ax_rms.plot(everyns, rmss, 'o', color=clr)

        ax_rms.set_xlabel('Every n')
        ax_rms.set_ylabel(r'RMS')
        #ax_rms.set_ylim(-1., 1.1*np.max(orig))


        ax_spec.plot(wv_grid, orig,  'o', color=clr, label=f'Tara: idx={idx}')
        # Fit
        ax_spec.plot(wv_grid, sv_fits[ifit],  ls='-', color=clr,
                     label=f'B-Spline: everyn={everyns[ifit]}')

        ax_spec.set_xlabel('Wavelength (nm)')
        ax_spec.set_ylabel(r'$a(\lambda)$')
        ax_spec.set_ylim(0., 1.1*np.max(orig))
        ax_spec.legend(fontsize=15.)
        

    # Finish
    for ax in [ax_rms, ax_spec]:
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

    # b-splines
    if flg & (2**1):
        fig_bspline_tara()


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