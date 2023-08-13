""" Methods to explore the Tara Oceans dataset, typically data driven"""

import numpy as np
import warnings

from oceancolor.tara import io 
from oceancolor.tara import  spectra

try:
    import sequencer
except ImportError:
    warnings.warn("sequencer not installed.  Some functions will not work.")
else:
    from sequencer import sequencer_

def prep_spectra(wv_grid:np.ndarray=None, min_sn:float=1.):

    if wv_grid is None:
        wv_grid = np.arange(402.5, 707.5, 5.) # nm

    # Load the data
    tara_db = io.load_tara_db()

    # Process to common wavelengths
    wv_nm, all_a_ph, all_a_ph_sig = spectra.spectra_from_table(tara_db)
    rwv_nm, r_aph, r_sig = spectra.rebin_to_grid(wv_nm, all_a_ph, all_a_ph_sig, wv_grid) 

    # Cull bad spectra
    tot_spec = np.nansum(r_aph, axis=-1)
    gd_tot = tot_spec > 0.

    # TODO
    # Cut on S/N
    med_sn = np.nanmedian(r_aph/r_sig, axis=1)
    cut_sn = med_sn > min_sn

    all_gd = gd_tot & cut_sn

    cull_raph = r_aph[all_gd, :]
    cull_rsig = r_sig[all_gd, :]

    # Deal with bad values
    really_bad = np.isnan(cull_raph) | (cull_rsig <= 0.) 
    # Replace
    cull_raph[really_bad] = 1e-5
    cull_rsig[really_bad] = 1e5

    # Negative
    negative = cull_raph < 0.
    cull_raph[negative] = 1e-5

    # Return
    return rwv_nm, cull_raph, cull_rsig

def run_sequencer(waves:np.ndarray, aph:np.ndarray, 
                  output_path:str,
                  estimator_list:list=None):

    # Init
    if estimator_list is None:
        estimator_list = ['EMD', 'energy', 'L2'] 

    grid = np.arange(len(waves))
    seq = sequencer.Sequencer(grid, aph, estimator_list)

    # Execute
    final_elongation, final_sequence = seq.execute(output_path)

    