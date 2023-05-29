""" Ingest the TARA database kindly provided by Ali Chase """

import os
import glob

#import numpy as np
#import seaborn as sns
#from matplotlib import pyplot as plt

import pandas

# HARD CODED FOR INGESTION ONLY
tara_path = '/home/xavier/Projects/Oceanography/Color/data/Tara'

from IPython import embed

def read_one_file(ofile:str):
    # Read
    f = open(ofile)
    lines = f.readlines()
    f.close()

    # Parse
    for line in lines:
        if line[0] != '/':
            continue
        # Fields?
        if 'fields' in line:
            fields = line.strip().split('=')[1].split(',')
        # Units
        if 'units' in line:
            units = line.strip().split('=')[1].split(',')
    # Ignore the dummy
    fields = fields

    # Pandas 
    df = pandas.read_table(ofile, comment='/', 
                           names=fields, 
                           delimiter=' ', index_col=False)

    # Add datetime
    df['datetime']  = pandas.to_datetime(
        [str(df['date'][idx]) + ' ' + df['time'][idx] for idx in range(len(df))], 
        format='%Y%m%d %H:%M:%S')

    # Drop the dummy
    df.drop(columns='', inplace=True)

    # Return
    return df, units

def load_cruise(cruise:str):
    files = glob.glob(os.path.join(tara_path, cruise, 
                                   f'Tara_ACS_*ap.txt'))

    # Loop me
    dfs = []
    for ifile in files:
        # ap
        df, units = read_one_file(ifile)
        # cp
        cp_file = ifile.replace('ap.txt', 'cp.txt')
        df_cp, _ = read_one_file(cp_file)
        # Drop columns
        df_cp.drop(columns=['date', 'time', 'lat', 'lon', 'Wt', 'sal'], inplace=True)
        # Merge
        df = df.merge(df_cp, on='datetime', suffixes=('_ap', '_cp'))
        embed(header='load_cruise')
        dfs.append(df)

    # Concatenate
    if len(dfs) > 0:
        df = pandas.concat(dfs, ignore_index=True)
    else:
        print(f"WARNING: No files found for {cruise}")
        return None

    # Add cruise name
    df['cruise'] = cruise

    # Return
    return df

def load_all():

    # Get the cruises
    cruises = [directory for directory in os.listdir(tara_path) 
               if os.path.isdir(os.path.join(tara_path,directory))]

    # Loop me
    dfs_ap, dfs_cp = [], []
    for cruise in cruises:
        print(f"Loading {cruise}...")
        # ap
        df = load_cruise(cruise, data=data)
        if df is None:
            continue
        # Append
        dfs.append(df)

    # Concatenate
    df = pandas.concat(dfs, ignore_index=True)

    embed(header='load_all')
    # Return
    return df

# Testing
if __name__ == '__main__':
    '''
    # One file
    ex_file = os.path.join(tara_path, 'CT-Rio', 'Tara_ACS_apcp2010_286cp.txt')
    df, units = read_one_file(ex_file)
    '''

    # One cruise
    load_cruise('CT-Rio')

    # All
    #df = load_all()