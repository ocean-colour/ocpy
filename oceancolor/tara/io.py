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

def load_pg_db():
    pg_db_name = os.path.join(resource_filename(
        'oceancolor', 'data'), 'Tara', 'merged_tara_pacific_microbiome_acs.feather')
    df = pandas.read_feather(pg_db_name)

    # Return
    return df