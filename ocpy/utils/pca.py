""" PCA methods for Ocean Color """

import numpy as np

from sklearn import decomposition

def fit_normal(data:np.ndarray, N:int, save_outputs:str=None,
               extra_arrays:dict=None):
    """ Fit a PCA to the data


    Args:
        data (np.ndarray): Data to fit
        N (int): Number of PCA components to fit
        save_outputs (str, optional): If provided, save the 
            fit items to a npz file. Defaults to None.
        extra_arrays (dict, optional): dict of extra arrays.
            If provided, save the extras to the npz file. Defaults to None.

    Returns:
        PCA: Fitted pca to the input data
    """


    # Fit
    pca_fit = decomposition.PCA(n_components=N).fit(data)

    # Save?
    if save_outputs:
        coeff = pca_fit.transform(data)
        #
        outputs = dict(Y=coeff,
                 data=data,
                 M=pca_fit.components_,
                 mean=pca_fit.mean_,
                 explained_variance=pca_fit.explained_variance_ratio_)
        if extra_arrays:
            outputs.update(extra_arrays)
        # Save
        np.savez(save_outputs, **outputs)
        print(f'Wrote: {save_outputs}')


    # All done
    return pca_fit

def reconstruct(pca_fit, vec):
    Y = pca_fit.transform(vec)
    recon = np.dot(Y, pca_fit.components_)
    # Return
    return recon[0] # Flatten
