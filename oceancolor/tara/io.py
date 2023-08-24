""" Methods for I/O on Tara Oceans data. """
import os

from pkg_resources import resource_filename
import pandas

from IPython import embed

db_name = os.path.join(resource_filename(
        'oceancolor', 'data'), 'Tara', 'Tara_APCP.parquet')

def load_tara_db():
    """ Load the Tara Oceans database. 

    Returns:
        pandas.DataFrame: table of data
    """
    # Get the file
    # Read
    df = pandas.read_parquet(db_name)

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