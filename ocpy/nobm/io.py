""" I/O functions for NOMB """

import os
import xarray

from IPython import embed

nobm_file = os.path.join(os.getenv('OS_COLOR'), 'data', 'NOBM', 
                         'all_NOBM_OASIM_data_2020.nc')


def load_nomb():
    nobm_xds = xarray.open_dataset(nobm_file, decode_times=False)

    # Unpack
    wave = nobm_xds.wavelength.data

    # Stack
    nobm_Rrs_stacked = nobm_xds.rrs.stack(z=("lat", "lon", "months"))
    nobm_a_stacked = nobm_xds.total_a.stack(z=("lat", "lon", "months"))
    nobm_bb_stacked = nobm_xds.total_bb.stack(z=("lat", "lon", "months"))

    # Remove nans
    nobm_Rrs_stacked = nobm_Rrs_stacked.dropna(dim='z')
    nobm_a_stacked = nobm_a_stacked.dropna(dim='z')
    nobm_bb_stacked = nobm_bb_stacked.dropna(dim='z')

    # Return
    return wave, nobm_Rrs_stacked.T, nobm_a_stacked.T, nobm_bb_stacked.T



'''
# Testing
if __name__ == '__main__':

    wave, rrs, a, bb = load_nomb()

    embed(header='33 of io.py: load_nomb() loaded the NOBM data.')
'''