""" Code to fit L23 with random Rs error """

import os

import numpy as np

from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from matplotlib import pyplot as plt
import seaborn as sns

import corner

from oceancolor.ihop import mcmc
from oceancolor.ihop import io as ihop_io
from oceancolor.ihop.nn import SimpleNet

from IPython import embed

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
        #p0=p0,
        save_file=pdict['save_file'])

    # Return
    return sampler, idx

def fit_fixed_perc(perc:int, n_cores:int, seed:int=1234,
                   Nspec:int=100):
    # Outfile
    outfile = os.path.join(
        os.getenv('OS_COLOR'), 'IHOP', 'L23',
        f'fit_a_L23_NN_Rs{perc:02d}')

    # Load Hydrolight
    print("Loading Hydrolight data")
    ab, Rs, d_a, d_bb = ihop_io.load_loisel_2023_pca()

    # Select a random sample
    np.random.seed(seed)
    idx = np.random.choice(np.arange(Rs.shape[0]), 
                           Nspec, replace=False)

    # Load NN
    model = ihop_io.load_nn('model_100000')

    # MCMC
    pdict = dict(model=model)
    pdict['nwalkers'] = 16
    pdict['nsteps'] = 10000
    pdict['save_file'] = None

    # Setup for parallel
    map_fn = partial(fit_one, pdict=pdict)

    # Prep
    items = [(Rs[i], ab[i], idx) for i in idx]
    
    # Parallel
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        chunksize = len(items) // n_cores if len(items) // n_cores > 0 else 1
        answers = list(tqdm(executor.map(map_fn, items,
                                            chunksize=chunksize), total=len(items)))

    embed(header='79 of figs')
    # Slurp
    samples = [item[0].get_chain() for item in answers]
    all_idx = np.array([item[1] for item in answers])
    srt = np.argsort(all_idx)

    # Chains
    all_samples = np.zeros((len(samples), samples[0].shape[0], 
        samples[0].shape[1], samples[0].shape[2]))
    for kk, ss in enumerate(srt):
        all_samples[ss,:,:,:] = samples[ss]


    

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
    another_test()
