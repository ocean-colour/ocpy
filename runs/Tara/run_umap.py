""" Run UMAP on Tara data """
import os
import numpy as np
import pickle

import pandas
import umap

from oceancolor.tara import explore

from IPython import embed

def run_umap(umap_tblfile:str, umap_savefile:str, process:dict=None):
    """ Run UMAP on a set of Tara spectra

    Args:
        umap_tblfile (str): Table holding the UMAP info
        umap_savefile (str): UMAP pickle file
        process (dict, optional): dict describing how to process the Tara data. Defaults to None.
    """

    # prep
    rwv_nm, cull_raph, cull_rsig, tara_tbl = explore.prep_spectra(process=process)

    # Train
    print("Training..")
    reducer_umap = umap.UMAP(random_state=42)
    latents_mapping = reducer_umap.fit(cull_raph)
    print("Done..")

    # Save?
    pickle.dump(latents_mapping, open(umap_savefile, "wb" ) )
    print(f"Saved UMAP to {umap_savefile}")
    embedding = latents_mapping.transform(cull_raph)

    # Generate table (only a few entries)
    umap_tbl = pandas.DataFrame()
    umap_tbl['tara_id'] = tara_tbl.index
    umap_tbl['U0'] = embedding[:,0]
    umap_tbl['U1'] = embedding[:,1]

    # Save
    umap_tbl.to_parquet(umap_tblfile)
    print(f"Wrote UMAP table to {umap_tblfile}")

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # UMAP me
    if flg & (2**0):

        # Un-normalized, i.e. Absolute
        out_tbl_file = os.path.join(
            os.getenv('OS_COLOR'), 'Tara', 'UMAP', 'Tara_UMAP_abs.parquet')
        umap_savefile = os.path.join(
            os.getenv('OS_COLOR'), 'Tara', 'UMAP', 'Tara_UMAP_abs.pkl')
        run_umap(out_tbl_file, umap_savefile)

    # UMAP with normalized spectra
    if flg & (2**1):

        # Normalize PDF
        out_tbl_file = os.path.join(
            os.getenv('OS_COLOR'), 'Tara', 'UMAP', 'Tara_UMAP_norm.parquet')
        umap_savefile = os.path.join(
            os.getenv('OS_COLOR'), 'Tara', 'UMAP', 'Tara_UMAP_norm.pkl')
        process = dict(Norm_PDF=True)
        run_umap(out_tbl_file, umap_savefile, process=process)




# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        flg += 2 ** 0  # 1 -- Unnormalied
        flg += 2 ** 1  # 2 -- Normalied
    else:
        flg = sys.argv[1]

    main(flg)
