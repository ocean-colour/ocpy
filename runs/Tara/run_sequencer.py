""" Run Sequencer on Tara data """
import numpy as np
import os

import pandas

from oceancolor.tara import explore

from sequencer import sequencer_

from IPython import embed

def run_sequencer(output_path:str, out_tbl_file:str,
                  estimator_list:list=None, process:dict=None,
                  nrand:int=None, norm:bool=False):

    # Init
    if estimator_list is None:
        estimator_list = ['EMD', 'energy', 'L2'] 

    # prep
    rwv_nm, cull_raph, cull_rsig, tara_tbl = explore.prep_spectra(process=process)

    # Random set?
    if nrand is not None:
        use_these = np.random.choice(np.arange(cull_raph.shape[0]), size=nrand, replace=False)
    else:
        use_these = np.arange(cull_raph.shape[0])

    # Cut
    cull_raph = cull_raph[use_these,:]
    tara_tbl = tara_tbl.iloc[use_these]

    # Init
    grid = np.arange(len(rwv_nm))
    seq = sequencer_.Sequencer(grid, cull_raph, estimator_list, no_norm=not norm)

    # Run
    final_elongation, final_sequence = seq.execute(output_path)

    # Some stats
    # print the resulting elongation
    print("resulting elongation: ", final_elongation)


    # print the intermediate elongations for different metrics + scales
    estimator_list, scale_list, elongation_list = seq.return_elongation_of_weighted_products_all_metrics_and_scales()


    print("intermediate elongations for the different metrics and scales:")
    for i in range(len(estimator_list)):
        print("metric=%s, scale=%s, elongation: %s" % (estimator_list[i], 
                                                    scale_list[i], 
                                                    np.round(elongation_list[i], 2)))

    # Keep the best model
    estimator_list, scale_list, sequence_list = seq.return_sequence_of_weighted_products_all_metrics_and_scales()
    best = np.argmax(elongation_list)
    final_sequence = sequence_list[best]

    # Write
    # Generate table (only a few entries)

    seq_tbl = pandas.DataFrame()
    seq_tbl['tara_id'] = tara_tbl.index[final_sequence]

    seq_tbl.to_parquet(os.path.join(output_path, out_tbl_file))
    print(f"Wrote: {os.path.join(output_path, out_tbl_file)}")

    return
    
def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Unnormalized
    if flg & (2**0):
        process = None
        output_path = os.path.join(
            os.getenv('OS_COLOR'), 'Tara', 'Sequencer', 'Abs')
        tbl_file = os.path.join(output_path,
            'Tara_Sequencer_abs.parquet')

        run_sequencer(output_path, tbl_file,
                      process=process,
                      estimator_list=['L2'],
                      nrand=10000)
                      #nrand=100)
        # 5 min for 10,0000 spectra for one metric scale=1


# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        flg += 2 ** 0  # 1 -- Unnormalied
    else:
        flg = sys.argv[1]

    main(flg)
