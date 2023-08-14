""" Run UMAP on Tara data """
import os
import numpy as np
import pickle

import umap

from oceancolor.tara import explore

from IPython import embed

def run_umap(umap_tblfile:str, umap_savefile:str=None):
    # Output files
    if umap_savefile is None:
        umap_savefile = os.path.join(
            os.getenv('OS_COLOR'), 'Tara', 'UMAP', 'Tara_UMAP.pkl')

    # prep
    rwv_nm, cull_raph, cull_rsig, tara_tbl = explore.prep_spectra()
    nspec = cull_raph.shape[0]

    embed(header='19 of run_umap')

    # Train
    reducer_umap = umap.UMAP(random_state=42)
    latents_mapping = reducer_umap.fit(cull_raph)
    print("Done..")

    # Save?
    pickle.dump(latents_mapping, open(umap_savefile, "wb" ) )
    print(f"Saved UMAP to {umap_savefile}")
    embedding = latents_mapping.transform(cull_raph)

    # Save table (only a few entries)

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # UMAP me
    if flg & (2**0):
        out_tbl_file = os.path.join(
            os.getenv('OS_COLOR'), 'Tara', 'UMAP', 'Tara_UMAP.parquet')
        run_umap(out_tbl_file)




# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        flg += 2 ** 0  # 1 -- Unnormalied
    else:
        flg = sys.argv[1]

    main(flg)
