""" Methods related to Tara Oceans data. """

import numpy as np
import pandas

import hashlib
import uuid

from IPython import embed


def tara_uid(df:pandas.DataFrame):
    """ Generate a UID for Tara Oceans data

    Done by concatenating the latitude and longitude to 4 decimal places.
    Modified in place

    Args:
        df (pandas.DataFrame): DataFrame of Tara Oceans data

    Returns:
        pandas.DataFrame: DataFrame with UID column
    """
    # Generate UID
    #time = [item.year*10000+item.month*100+item.day for item in df.index]
    #uid = ((180.+df.lon.values)*100000).astype(np.uint64)*1000000000000000+ \
    #    ((df.lat.values+90.)*100000).astype(np.uint64)*100000000 \
    #        + np.array(time).astype(np.uint64)
    return
    embed(header='tara_utils 28')
    uid = [hashlib.sha256(f'{item.lat:.5f}{item.lon:.5f}{item.t:.6f}'.encode()).hexdigest() for item in df.itertuples()]

    # Check for duplicates
    if len(uid) != len(np.unique(uid)):
        embed(header='tara_utils 26')
        raise ValueError("Duplicate UID")
    # Add to DataFrame
    df['UID'] = uid