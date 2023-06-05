""" Methods for I/O on Tara Oceans data. """
import os

from pkg_resources import resource_filename
import pandas

from IPython import embed

db_name = os.path.join(resource_filename(
        'oceancolor', 'data'), 'Tara', 'Tara_APCP.parquet')

def load_tara_db():
    """ Load the Tara Oceans database. """
    # Get the file
    # Read
    df = pandas.read_parquet(db_name)
    # Return
    return df
