""" Methods related to I/O """

import os
import json
import gzip

import numpy as np

from IPython import embed


def jsonify(obj, debug=True):
    """ Recursively process an object so it can be serialised in json
    format.

    WARNING - the input object may be modified if it's a dictionary or
    list!

    Parameters
    ----------
    obj : any object
    debug : bool, optional

    Returns
    -------
    obj - the same obj is json_friendly format (arrays turned to
    lists, np.int64 converted to int, np.float64 to float, and so on).

    """
    if isinstance(obj, np.float64):
        obj = float(obj)
    elif isinstance(obj, np.float32):
        obj = float(obj)
    elif isinstance(obj, np.int32):
        obj = int(obj)
    elif isinstance(obj, np.int64):
        obj = int(obj)
    elif isinstance(obj, np.int16):
        obj = int(obj)
    elif isinstance(obj, np.bool_):
        obj = bool(obj)
    elif isinstance(obj, np.bytes_):
        obj = str(obj)
    elif isinstance(obj, np.ndarray):  # Must come after Quantity
        obj = obj.tolist()
    elif isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = jsonify(value, debug=debug)
    elif isinstance(obj, list):
        for i,item in enumerate(obj):
            obj[i] = jsonify(item, debug=debug)
    elif isinstance(obj, tuple):
        obj = list(obj)
        for i,item in enumerate(obj):
            obj[i] = jsonify(item, debug=debug)
        obj = tuple(obj)
    #else:
    #    if debug:
    #        embed(header='72 of ')
            

    if debug:
        #print(type(obj))
        # TODO - add None type
        if not isinstance(obj, (list,dict,str,int,float,type(None))):
            print(f"Bad type: {type(obj)}")
            embed(header='79 of io.py')

    return obj


def loadjson(filename):
    """
    Parameters
    ----------
    filename : str

    Returns
    -------
    obj : dict

    """
    #
    if filename.endswith('.gz'):
        with gzip.open(filename, "rb") as f:
            obj = json.loads(f.read().decode("ascii"))
    else:
        with open(filename, 'rt') as fh:
            obj = json.load(fh)

    return obj



def savejson(filename, obj, overwrite=False, indent=None, easy_to_read=False,
             **kwargs):
    """ Save a python object to filename using the JSON encoder.

    Parameters
    ----------
    filename : str
    obj : object
      Frequently a dict
    overwrite : bool, optional
    indent : int, optional
      Input to json.dump
    easy_to_read : bool, optional
      Another approach and obj must be a dict
    kwargs : optional
      Passed to json.dump

    Returns
    -------

    """
    import io

    if os.path.lexists(filename) and not overwrite:
        raise IOError('%s exists' % filename)
    if easy_to_read:
        if not isinstance(obj, dict):
            raise IOError("This approach requires obj to be a dict")
        with io.open(filename, 'w', encoding='utf-8') as f:
            f.write(json.dumps(obj, sort_keys=True, indent=4,
                               separators=(',', ': '), **kwargs))
    else:
        if filename.endswith('.gz'):
            with gzip.open(filename, 'wt') as fh:
                json.dump(obj, fh, indent=indent, **kwargs)
        else:
            with open(filename, 'wt') as fh:
                json.dump(obj, fh, indent=indent, **kwargs)