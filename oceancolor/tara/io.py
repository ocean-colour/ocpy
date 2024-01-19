""" Methods for I/O on Tara Oceans data. """
import os

import numpy as np
import warnings

from pkg_resources import resource_filename
import pandas

try:
    import geopandas
except ImportError:
    warnings.warn("geopandas not installed")

from IPython import embed


def load_db(dataset='pg'):
    """
    Load the specified Tara dataset.

    Args:
        dataset (str): The dataset to load. Options are 'pg' for 
        Patrick Gray database and 'ac' for Alison Chase database. 
        Default is 'pg'.

    Returns:
        DataFrame: The loaded dataset as a pandas DataFrame.
    """

    if dataset == 'pg':
        df = load_pg_db()
    elif dataset == 'ac':
        df = load_ac_db()

    # Return
    return df

def load_ac_db():
    """ Load the Tara Oceans database kindly provided by Alison Chase.

    TODO -- Add publication info

    Returns:
        pandas.DataFrame: table of data
    """
    # Get the file
    db_name = os.path.join(resource_filename(
        'oceancolor', 'data'), 'Tara', 'Tara_APCP.parquet')
    # Read
    df = pandas.read_parquet(db_name)

    # Return
    return df

def load_pg_db(expedition:str='all', as_geo:bool=False):
    """
    Load the Tara Oceans database from a feather file.

    Args:
        expedition (str): The expedition to load. Options are 'all', 'Microbiome', or 'Pacific'. Default is 'all'.
        as_geo (bool): If True, load the database as a geopandas DataFrame. Default is False.

    Returns:
        pandas.DataFrame or geopandas.GeoDataFrame: The loaded database.
    """
    pg_db_name = os.path.join(resource_filename(
        'oceancolor', 'data'), 'Tara', 'merged_tara_pacific_microbiome_acs_160124.feather')
        #'oceancolor', 'data'), 'Tara', 'merged_tara_pacific_microbiome_acs.feather') # Old file
    print(f"Reading: {pg_db_name}")
    if as_geo:
        df = geopandas.read_feather(pg_db_name)
    else:
        df = pandas.read_feather(pg_db_name)

    # Check on unique times
    times = df.index.astype(int)
    uni, idx = np.unique(times, return_index=True)
    if len(uni) != len(times):
        #raise ValueError("Duplicate times in Tara Oceans database")
        warnings.warn("Duplicate times in Tara Oceans database")
        df = df.iloc[idx,:]

    # Add ID number?
    if 'uid' not in df.columns:
        df['uid'] = times[idx]

    # Cut?
    if expedition == 'Microbiome':
        df = df[df.index > pandas.Timestamp('2020-01-01')]
    elif expedition == 'all':
        pass
    elif expedition == 'Pacific':
        df = df[df.index < pandas.Timestamp('2020-01-01')]
    else:
        raise ValueError(f"Bad mission: {expedition}")

    # Return
    return df

def load_tara_umap(utype:str):

    # Load UMAP table
    umap_file = os.path.join(
        os.getenv('OS_COLOR'), 'Tara', 'UMAP', f'Tara_UMAP_{utype}.parquet')
    if not os.path.exists(umap_file):
        raise ValueError(f"Bad utype: {utype}")
    umap_tbl = pandas.read_parquet(umap_file)

    # Return
    return umap_tbl

def load_tara_sequencer(utype:str):
    if utype == 'abs':
        tbl_file = os.path.join(os.getenv('OS_COLOR'), 'Tara', 'Sequencer',
                            'Abs', 'Tara_Sequencer_abs.parquet')
    elif utype == 'norm':
        tbl_file = os.path.join(os.getenv('OS_COLOR'), 'Tara', 'Sequencer',
                            'Norm', 'Tara_Sequencer_norm.parquet')
                        
    tara_seq = pandas.read_parquet(tbl_file)

    # Return
    return tara_seq
