""" Code to fit L23 with random Rs error """

import os

from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from oceancolor.ihop import mcmc
from oceancolor.ihop import io as ihop_io

from IPython import embed

def fit_one(items:list, pdict:dict=None):
    # Unpack
    Rs, ab_pca = items

    # Init (cheating, but do it)
    ndim = nn_model.ninput
    p0 = np.outer(np.ones(pdict['nwalkers']), 
                  ab_pca)

    # Run
    sampler = mcmc.run_emcee_nn(
        pdict['model'], Rs,
        nwalkers=pdict['nwalkers'],
        nsteps=pdict['nsteps'],
        p0=p0,
        save_file=pdict['save_file'])

    # Return
    return sampler

def fit_fixed_perc(perc:int, n_cores:int):
    # Outfile
    outfile = os.path.join(
        os.getenv('OS_COLOR'), 'IHOP', 'L23',
        f'fit_a_L23_NN_Rs{perc:02d}')

    # Load NN
    model = ihop_io.load_nn('model_100000')
    pdict = dict(model=model)

    # MCMC
    pdict['nwalkers'] = 16
    pdict['nsteps'] = 10000
    pdict['save_file'] = None

    # Setup for parallel
    map_fn = partial(fit_one, pdict=pdict)

    # Prep
    items = [item for item in zip(U_fields,V_fields,sub_idx)]
    
    # Parallel
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        chunksize = len(items) // n_cores if len(items) // n_cores > 0 else 1
        answers = list(tqdm(executor.map(map_fn, items,
                                            chunksize=chunksize), total=len(items)))
    # Slurp
    img_idx += [item[0] for item in answers]
    pp_fields += [item[1] for item in answers]

def quick_test():
    # Load Hydrolight
    print("Loading Hydrolight data")
    ab, Rs, d_l23 = ihop_io.load_loisel_2023_pca()

    # Load model
    model = ihop_io.load_nn('model_100000')

    pdict['nwalkers'] = 16
    pdict['nsteps'] = 10000
    pdict['save_file'] = 'tmp.h5'

    items = [Rs, ab]

    sampler = fit_one(items, pdict=pdict)

    embed(header='80 quick test')

if __name__ == '__main__':
    quick_test()
