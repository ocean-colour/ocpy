""" PCA this and that in Remote Sensing """

import os
import numpy as np

from importlib import resources

from oceancolor.hydrolight import loisel23
from oceancolor import pca
from oceancolor.tara import io as tara_io
from oceancolor.tara import spectra
from oceancolor import water
from oceancolor.utils import spectra as spec_utils 

from IPython import embed

pca_path = os.path.join(resources.files('oceancolor'),
                            'data', 'PCA')



def generate_all_pca(clobber:bool=False):
    generate_l23_pca(clobber=clobber)
    generate_l23_tara_pca(clobber=clobber)


def generate_l23_pca(clobber:bool=False):

    # Load up the data
    X=4
    Y=0
    ds = loisel23.load_ds(X, Y)

    # L23, vanilla
    for iop in ['a', 'b', 'bb']:
        outfile = os.path.join(pca_path, f'pca_L23_X4Y0_{iop}_N3.npz')
        if not os.path.exists(outfile) or clobber:
            pca.fit_normal(ds[iop].data, 3, save_outputs=outfile)

    # L23 positive, definite

def generate_l23_tara_pca(clobber:bool=False):

    # Load up the data
    X=4
    Y=0
    l23_ds = loisel23.load_ds(X, Y)

    # Load up Tara
    print("Loading Tara..")
    tara_db = tara_io.load_tara_db()
    # Spectra
    wv_nm, all_a_p, all_a_p_sig = spectra.spectra_from_table(tara_db)

    # Wavelengths, restricted to > 400 nm
    cut = l23_ds.Lambda > 400.
    l23_a = l23_ds.a.data[:,cut]

    # Rebin
    wv_grid = l23_ds.Lambda.data[cut]
    tara_wv = np.append(wv_grid, [755.]) - 2.5 # Because the rebinning is not interpolation
    rwv_nm, r_ap, r_sig = spectra.rebin_to_grid(
        wv_nm, all_a_p, all_a_p_sig, tara_wv)

    # Add in water
    print("Adding in water..")
    df_water = water.load_rsr_gsfc()
    a_w, _ =  spec_utils.rebin(df_water.wavelength.values, 
                        df_water.aw.values, np.zeros_like(df_water.aw),
                        wv_grid)

    tara_a_water = r_ap+np.outer(np.ones(r_ap.shape[0]), a_w)

    # Polish Tara for PCA
    bad = np.isnan(tara_a_water) | (tara_a_water <= 0.)

    # Replace
    tara_a_water[bad] = 0.
    tara_a_water[bad] = 1e5

    # N components
    data = np.append(l23_a, tara_a_water, axis=0)
    for N in [3,20]:
        print(f"Fit PCA with N={N}")
        outfile = os.path.join(pca_path, f'pca_L23_X4Y0_Tara_a_N{N}.npz')
        if not os.path.exists(outfile) or clobber:
            pca.fit_normal(data, N, save_outputs=outfile)


def load_pca(pca_file:str):
    return np.load(os.path.join(pca_path, pca_file))

if __name__ == '__main__':
    generate_all_pca()