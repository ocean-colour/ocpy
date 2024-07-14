""" Methods related to the GLORIA dataset

https://www.nature.com/articles/s41597-023-01973-y
https://doi.org/10.1594/PANGAEA.948492 

"""
import os
import pandas
import numpy as np
from importlib import resources

def load_gloria():
    """
    Load the GLORIA dataset

    Returns:
        tuple: A tuple containing the following dataframes:
            df_meta (pandas.DataFrame): Metadata and lab data
            df_Rrs_mean (pandas.DataFrame): Mean Rrs values
            df_Rrs_std (pandas.DataFrame): Standard deviation of Rrs values
            df_qc_flags (pandas.DataFrame): Quality control flags
    """
    print("Loading GLORIA dataset...")
    gloria_path = os.path.join(resources.files('ocpy'), 
                                 'data', 'Rrs', 'GLORIA') 
    # Files
    Rrs_file = os.path.join(gloria_path, 'GLORIA_Rrs.csv')
    Rrs_std_file = os.path.join(gloria_path, 'GLORIA_Rrs_std.csv')
    qc_flags_file = os.path.join(gloria_path, 'GLORIA_qc_flags.csv')
    meta_file = os.path.join(gloria_path, 'GLORIA_meta_and_lab.csv')
                                
    # Load
    df_Rrs_mean = pandas.read_csv(Rrs_file)
    df_Rrs_std = pandas.read_csv(Rrs_std_file)
    df_qc_flags = pandas.read_csv(qc_flags_file)
    df_meta = pandas.read_csv(meta_file)

    # Return
    return df_meta, df_Rrs_mean, df_Rrs_std, df_qc_flags


def parse_table(df, flavor:str):
    """ Parse wavelengths and values
    from a row/table of the GLORIA database. 

    Args:
        df (pandas.DataFrame):
            One row or table of the Tara Oceans database.
        flavor (str):
            Flavor of table

    Returns:
        tuple: 
            wavelengths (nm) [np.ndarray], 
            values [np.ndarray] (nwv, nspec) 
            keys [np.ndarray]
    """
    keys, wv_nm = [], []

    values = np.zeros((5000, len(df)))

    ss = 0
    for key in df.keys():
        if key[:len(flavor)] == flavor:
            # Keep
            keys.append(key)
            # Wavelength
            wv_nm.append(float(key[len(flavor)+1:]))
            # Values
            values[ss,:] = df[key].values
            ss += 1

    # Recast
    wv_nm = np.array(wv_nm)
    keys = np.array(keys)

    # Return
    return wv_nm, values[:ss,:], keys