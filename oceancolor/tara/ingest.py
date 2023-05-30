""" Ingest the TARA database kindly provided by Ali Chase """

import os
import glob
import warnings

#import numpy as np
#import seaborn as sns
#from matplotlib import pyplot as plt

import pandas

# HARD CODED FOR INGESTION ONLY
tara_path = '/home/xavier/Projects/Oceanography/Color/data/Tara'
drop_columns = ['date', 'time', 'lat', 'lon', 'Wt', 'sal']

from IPython import embed

warnings.warn('ADD in META!')

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
    df_val = pandas.read_table(ofile, comment='/', 
                           names=fields, 
                           delimiter=' ', index_col=False)
    sig_file = ofile.replace('.txt', '_uncertainty.txt')

    if os.path.isfile(sig_file):
        df_sig = pandas.read_table(sig_file, comment='/', 
                           names=fields, 
                           delimiter=' ', index_col=False)
    else:
        warnings.warn(f"No uncertainty file for {ofile}")
        df_sig = None

    # Add datetime
    for df in [df_val, df_sig]:
        if df is None:
            continue
        df['datetime']  = pandas.to_datetime(
            [str(df['date'][idx]) + ' ' + df['time'][idx] for idx in range(len(df))], 
            format='%Y%m%d %H:%M:%S')
        df.drop(columns='', inplace=True)

    # Rename sig
    if df_sig is None:
        df = df_val.copy()
    else:
        rename_dict = {}
        for key in df_sig.keys():
            if 'ap' in key or 'cp' in key:
                rename_dict[key] = f'sig_{key}'
        df_sig.rename(columns=rename_dict, inplace=True)

        # Merge
        df_sig.drop(columns=drop_columns, inplace=True)
        df = df_val.merge(df_sig, on='datetime')

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
        if os.path.isfile(cp_file):
            df_cp, _ = read_one_file(cp_file)
            # Drop columns
            df_cp.drop(columns=drop_columns, inplace=True)
            # Merge
            df = df.merge(df_cp, on='datetime', suffixes=('_ap', '_cp'))
        else:
            warnings.warn(f"No cp file for {ifile}")

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
    dfs = []
    for cruise in cruises:
        print(f"Loading {cruise}...")
        # ap
        df = load_cruise(cruise)
        if df is None:
            continue
        # Append
        dfs.append(df)

    # Concatenate
    df = pandas.concat(dfs, ignore_index=True)

    # Return
    return df

# Testing
if __name__ == '__main__':
    # One file
    #ex_file = os.path.join(tara_path, 'CT-Rio', 
    #                       'Tara_ACS_apcp2010_286cp.txt')
    #df, units = read_one_file(ex_file)

    # One cruise
    #df = load_cruise('CT-Rio')
    #embed(header='ingest testing 129')

    # All
    df = load_all()
    embed(header='load_all 151')