""" Code to fit L23 with random Rs error """

import os

import numpy as np

from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import torch

from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200

import seaborn as sns

import corner

from oceancolor.ihop import mcmc
from oceancolor.ihop import io as ihop_io
from oceancolor.ihop.nn import SimpleNet
from oceancolor.ihop import pca as ihop_pca

from IPython import embed

out_path = os.path.join(
        os.getenv('OS_COLOR'), 'IHOP', 'L23')

def do_all_fits(n_cores:int=4):

    for perc in [0, 5, 10, 15, 20]:
        print(f"Working on: perc={perc}")
        fit_fixed_perc(perc=perc, n_cores=n_cores, Nspec=100)

def analyze_l23(chain_file, chop_burn:int=-4000):

    # #############################################
    # Load

    # Load Hydrolight
    print("Loading Hydrolight data")
    ab, Rs, d_a, d_bb = ihop_io.load_loisel_2023_pca()

    # MCMC
    print("Loading MCMC")
    d = np.load(os.path.join(out_path, chain_file))
    chains = d['chains']
    l23_idx = d['idx']

    all_medchi, all_stdchi, all_rms, all_maxdev = [], [], [], []
    all_mxwv = []
    
    # Loop
    for ss, idx in enumerate(l23_idx):
        # a
        Y = chains[ss, chop_burn:, :, 0:3].reshape(-1,3)
        orig, a_recon = ihop_pca.reconstruct(Y, d_a, idx)
        a_mean = np.mean(a_recon, axis=0)
        a_std = np.std(a_recon, axis=0)

        # Stats
        rms = np.std(a_mean-orig)
        chi = np.abs(a_mean-orig)/a_std
        dev = np.abs(a_mean-orig)/a_mean
        imax_dev = np.argmax(dev)
        max_dev = dev[imax_dev]
        mxwv = d_a['wavelength'][imax_dev]

        # Save
        all_rms.append(rms)
        all_maxdev.append(max_dev)
        all_medchi.append(np.median(chi))
        all_stdchi.append(np.std(chi))
        all_mxwv.append(mxwv)

    # Return
    stats = dict(rms=all_rms,
                 max_dev=all_maxdev,
                 med_chi=all_medchi,
                 std_chi=all_stdchi,
                 mx_wave=all_mxwv)
    return stats


def check_one(chain_file:str, in_idx:int, chop_burn:int=-4000):

    # #############################################
    # Load

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ihop_io.load_nn('model_100000')

    # Load Hydrolight
    print("Loading Hydrolight data")
    ab, Rs, d_a, d_bb = ihop_io.load_loisel_2023_pca()

    # MCMC
    print("Loading MCMC")
    d = np.load(os.path.join(out_path, chain_file))
    chains = d['chains']
    l23_idx = d['idx']
    obs_Rs = d['obs_Rs']

    idx = l23_idx[in_idx]
    # a
    Y = chains[in_idx, chop_burn:, :, 0:3].reshape(-1,3)
    orig, a_recon = ihop_pca.reconstruct(Y, d_a, idx)
    a_mean = np.mean(a_recon, axis=0)
    a_std = np.std(a_recon, axis=0)

    allY = chains[in_idx, chop_burn:, :, :].reshape(-1,6)
    all_pred = np.zeros((allY.shape[0], 81))
    for kk in range(allY.shape[0]):
        Ys = allY[kk]
        pred_Rs = model.prediction(Ys, device)
        all_pred[kk,:] = pred_Rs

    pred_Rs = np.median(all_pred, axis=0)
    std_pred = np.std(all_pred, axis=0)

    # #########################################################
    # Plot the solution
    plt.clf()
    ax = plt.gca()
    ax.plot(d_a['wavelength'], orig, 'ko', label='True')
    ax.plot(d_a['wavelength'], a_mean, 'r-', label='Fit')
    ax.fill_between(
        d_a['wavelength'], a_mean-a_std, a_mean+a_std, 
        color='r', alpha=0.5) 

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel(r'$a(\lambda)$')

    plt.show()

    '''
    # #########################################################
    # Plot the residuals
    plt.clf()
    ax = plt.gca()
    ax.plot(d_a['wavelength'], a_mean-orig, 'bo', label='True')
    ax.fill_between(d_a['wavelength'], -a_std, a_std, 
        color='r', alpha=0.5) 

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel(r'$a(\lambda)$ [Fit-Orig]')

    plt.show()
    '''

    # #########################################################
    # Compare Rs
    plt.clf()
    ax = plt.gca()
    ax.plot(d_a['wavelength'], Rs[idx], 'bo', label='True')
    ax.plot(d_a['wavelength'], obs_Rs[in_idx], 'ks', label='Obs')
    ax.plot(d_a['wavelength'], pred_Rs, 'rx', label='Model')

    ax.fill_between(
        d_a['wavelength'], pred_Rs-std_pred, pred_Rs+std_pred,
        color='r', alpha=0.5) 

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel(r'$R_s$')

    ax.legend()

    plt.show()

    

def fit_one(items:list, pdict:dict=None):
    # Unpack
    Rs, ab_pca, idx = items

    # Init (cheating, but do it)
    ndim = pdict['model'].ninput
    #p0 = np.outer(np.ones(pdict['nwalkers']), 
    #              ab_pca)

    # Run
    sampler = mcmc.run_emcee_nn(
        pdict['model'], Rs,
        nwalkers=pdict['nwalkers'],
        nsteps=pdict['nsteps'],
        scl_sig=pdict['perc']/100.,
        #p0=p0,
        save_file=pdict['save_file'])

    # Return
    return sampler, idx

def fit_fixed_perc(perc:int, n_cores:int, seed:int=1234,
                   Nspec:int=100):
    # Outfile
    outfile = os.path.join(out_path,
        f'fit_a_L23_NN_Rs{perc:02d}')

    # Load Hydrolight
    print("Loading Hydrolight data")
    ab, Rs, d_a, d_bb = ihop_io.load_loisel_2023_pca()

    # Select a random sample
    np.random.seed(seed)
    idx = np.random.choice(np.arange(Rs.shape[0]), 
                           Nspec, replace=False)

    # Add in random noise
    r_sig = np.random.normal(size=Rs.shape)
    r_sig = np.minimum(r_sig, 3.)
    r_sig = np.maximum(r_sig, -3.)
    Rs += (perc/100.) * Rs * r_sig

    # Load NN
    model = ihop_io.load_nn('model_100000')

    # MCMC
    pdict = dict(model=model)
    pdict['nwalkers'] = 16
    pdict['nsteps'] = 10000
    pdict['save_file'] = None
    pdict['perc'] = perc

    # Setup for parallel
    map_fn = partial(fit_one, pdict=pdict)

    # Prep
    items = [(Rs[i], ab[i], i) for i in idx]
    
    # Parallel
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        chunksize = len(items) // n_cores if len(items) // n_cores > 0 else 1
        answers = list(tqdm(executor.map(map_fn, items,
                                            chunksize=chunksize), total=len(items)))

    # Slurp
    samples = [item[0].get_chain() for item in answers]
    all_idx = np.array([item[1] for item in answers])

    # Chains
    all_samples = np.zeros((len(samples), samples[0].shape[0], 
        samples[0].shape[1], samples[0].shape[2]))
    for ss in range(len(all_idx)):
        all_samples[ss,:,:,:] = samples[ss]

    # Save
    np.savez(outfile, chains=all_samples, idx=all_idx,
             obs_Rs=Rs[all_idx])
    print(f"Wrote: {outfile}")

def another_test():
    fit_fixed_perc(perc=10, n_cores=4, Nspec=8)

def quick_test():
    # Load Hydrolight
    print("Loading Hydrolight data")
    ab, Rs, d_a, d_bb = ihop_io.load_loisel_2023_pca()

    # Load model
    model = ihop_io.load_nn('model_100000')

    pdict = dict(model=model)
    pdict['nwalkers'] = 16
    pdict['nsteps'] = 20000
    pdict['save_file'] = 'tmp.h5'

    idx = 1000
    items = [Rs[idx], ab[idx]]

    sampler = fit_one(items, pdict=pdict)
    samples = sampler.get_chain()
    samples = samples[-4000:,:,:].reshape((-1, samples.shape[-1]))

    cidx = 0
    plt.clf()
    plt.hist(samples[:,cidx], 100, color='k', histtype='step')
    ax = plt.gca()
    ax.axvline(ab[idx][cidx])
    #
    plt.show()

    # Corner
    fig = corner.corner(samples, labels=['a0', 'a1', 'a2', 'b0', 'b1', 'b2'],
                    truths=ab[idx])
    plt.show()

    # Plot
    embed(header='80 quick test')

if __name__ == '__main__':

    # Testing
    #quick_test()
    #another_test()

    # All of em
    #do_all_fits()

    # Analysis
    #stats = analyze_l23('fit_a_L23_NN_Rs10.npz')
    check_one('fit_a_L23_NN_Rs10.npz', 0)
