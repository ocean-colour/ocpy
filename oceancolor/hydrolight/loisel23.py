""" Loisel+2023 Hydrolight Outputs """

import os
import xarray

l23_path = os.path.join(os.getenv('OS_COLOR'), 
                        'data', 'Loisel2023')

def load_ds(X:int, Y:int): 

    # Load up the data
    variable_file = os.path.join(l23_path, 
                                 f'Hydrolight{X}{Y:02d}.nc')
    ds = xarray.load_dataset(variable_file)

    # Return
    return ds