""" Methods for I/O on Tara Oceans data. """
import os

from pkg_resources import resource_filename
import pandas

from IPython import embed

db_name = os.path.join(resource_filename(
        'oceancolor', 'data'), 'Tara', 'Tara_APCP.parquet')

def load_db(dataset='pg'):

    if dataset == 'pg':
        df = load_pg_db()
    elif dataset == 'ac':
        df = load_ac_db()

    # Return
    return df

def load_ac_db():
    """ Load the Tara Oceans database. 

    Kindly provided by Alison Chase.

    Returns:
        pandas.DataFrame: table of data
    """
    # Get the file
    # Read
    df = pandas.read_parquet(db_name)

    # Return
    return df

def load_pg_db(expedition:str='all'):
    pg_db_name = os.path.join(resource_filename(
        'oceancolor', 'data'), 'Tara', 'merged_tara_pacific_microbiome_acs.feather')
    df = pandas.read_feather(pg_db_name)

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
