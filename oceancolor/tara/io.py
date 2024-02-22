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

def load_pg_db(expedition:str='all', as_geo:bool=False, 
               clean_flagged:bool=True):
    """
    Load the Tara Oceans database from a feather file.

    Args:
        expedition (str): The expedition to load. Options are 'all', 'Microbiome', or 'Pacific'. Default is 'all'.
        as_geo (bool): If True, load the database as a geopandas DataFrame. Default is False.
        clean_flagged (bool): If True, remove flagged data. Default is True.

    Returns:
        pandas.DataFrame or geopandas.GeoDataFrame: The loaded database.
    """
    pg_db_name = os.path.join(resource_filename(
        'oceancolor', 'data'), 'Tara', 'merged_tara_pacific_microbiome_acs_160124.feather')
        #'oceancolor', 'data'), 'Tara', 'merged_tara_pacific_microbiome_acs.feather') # Old file
    print(f"Reading: {pg_db_name}")
    if as_geo:
        gdf = geopandas.read_feather(pg_db_name)
    else:
        gdf = pandas.read_feather(pg_db_name)

    # Check on unique times
    times = gdf.index.astype(int)
    uni, idx = np.unique(times, return_index=True)
    if len(uni) != len(times):
        #raise ValueError("Duplicate times in Tara Oceans database")
        warnings.warn("Duplicate times in Tara Oceans database")
        gdf = gdf.iloc[idx,:]

    # Add ID number?
    if 'uid' not in gdf.columns:
        gdf['uid'] = times[idx]


    def is_set(x, n):
        return x & 1 << n != 0

    # Mission
    gdf['mission_id'] = 0
    gdf.loc[gdf['datetime'] > '2020-01-01', 'mission_id'] = 1

    gdf['passes_flags'] = True

    # find relevant flags in TaraPacific
    passes_flag = []
    flag_bits = [3]

    for i in gdf[gdf.mission_id==0].flag_bit:
        passed = True
        for bit in flag_bits:
            if is_set(i,bit):
                passed = False
        passes_flag.append(passed)
        
    passes_flag = np.array(passes_flag, dtype=bool)

    gdf.loc[gdf['mission_id'] == 0, 'passes_flags'] = passes_flag

    # find relevant flags in TaraMicrobiome
    passes_flag = []
    flag_bits = [8,9]

    for i in gdf[gdf.mission_id==1].flag_bit:
        passed = True
        for bit in flag_bits:
            if is_set(i,bit):
                passed = False
        passes_flag.append(passed)
        
    passes_flag = np.array(passes_flag, dtype=bool)

    gdf.loc[gdf['mission_id'] == 1, 'passes_flags'] = passes_flag


    # dataframe with the flagged data removed
    if clean_flagged:
        print(f"Using bit_flags removes {np.sum(~gdf.passes_flags)} rows of a total {len(gdf)}")
        gdf = gdf[gdf.passes_flags]

    # Cut?
    if expedition == 'Microbiome':
        gdf = gdf[gdf.mission_id == 1]
    elif expedition == 'all':
        pass
    elif expedition == 'Pacific':
        gdf = gdf[gdf.mission_id == 0]
    else:
        raise ValueError(f"Bad mission: {expedition}")

    # Return
    return gdf

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
