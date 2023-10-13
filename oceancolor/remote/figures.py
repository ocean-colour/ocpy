""" Generate figures for the remote sensing project """

import os
import xarray
import numpy as np

from matplotlib import pyplot as plt

import emcee

from oceancolor.remote import io as remote_io

from IPython import embed

def fig_pca_mcmc(outfile:str, l23_idx:int, X:int=4, Y:int=0):

    # Load Hydrolight
    l23_path = os.path.join(os.getenv('OS_COLOR'),
                            'data', 'Loisel2023')
    variable_file = os.path.join(l23_path, 
                                 f'Hydrolight{X}{Y:02d}.nc')
    ds = xarray.load_dataset(variable_file)

    # And its PCA
    print("Loading Hydrolight data")
    ab, Rs, d_l23 = remote_io.load_loisel_2023_pca()

    # Load MCMC
    mcmc_file = f'MCMC_NN_i{l23_idx}.h5'
    reader = emcee.backends.HDFBackend(mcmc_file, read_only=True)
    flatchain = reader.get_chain(flat=True)

    # Cut down
    flatchain = flatchain[-10000:]

    # Generate the predictions
    a_recon = np.dot(flatchain[:,0:3], d_l23['a_M3']) + d_l23['a_mean']
    a_mean = np.mean(a_recon, axis=0)
    a_std = np.std(a_recon, axis=0)

    a_pca = np.dot(ab[l23_idx, 0:3], d_l23['a_M3']) + d_l23['a_mean']

    # Init plot
    fig = plt.figure(figsize=(12,7))
    axes = fig.subplots(nrows=1, ncols=2)

    # a(lambda)
    ax_a = axes[0]

    # Plot real answer
    ax_a.plot(ds.Lambda, ds.a.data[l23_idx,:], 'k-', label='True a')

    # Plot prediction
    ax_a.plot(ds.Lambda, a_mean, 'b--', label='Predicted a')

    # Plot PCA
    #ax_a.plot(ds.Lambda, a_pca, 'r:', label='PCA a')

    # Error interval
    ax_a.fill_between(ds.Lambda, a_mean-a_std, a_mean+a_std, 
                      color='b', alpha=0.2)

    ax_a.set_ylabel(r'$a(\lambda)$')
    ax_a.set_xlabel(r'$\lambda$ [nm]')
    ax_a.legend()

    # a(lambda) zoom-in
    ax_b = axes[1]

    # Plot real answer
    ax_b.plot(ds.Lambda, ds.a.data[l23_idx,:], 'k-', label='True a')

    # Plot prediction
    ax_b.plot(ds.Lambda, a_mean, 'b--', label='Predicted a')

    # Plot PCA
    #ax_a.plot(ds.Lambda, a_pca, 'r:', label='PCA a')

    # Error interval
    ax_b.fill_between(ds.Lambda, a_mean-a_std, a_mean+a_std, 
                      color='b', alpha=0.2)

    ax_b.set_ylabel(r'$a(\lambda)$')
    ax_b.set_xlabel(r'$\lambda$ [nm]')

    ax_b.set_xlim(350., 500)
    ax_b.set_ylim(0., 0.2)


    ax_a.legend()

    plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f'Wrote: {outfile}')

if __name__ == '__main__':
    fig_pca_mcmc('fig_pca_mcmc_l23_i200.png', 200)