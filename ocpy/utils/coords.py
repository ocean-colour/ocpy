""" Utilities related to coordinates. """

import numpy as np
from math import radians

from sklearn.metrics.pairwise import haversine_distances

import re

def dms_to_decimal(degrees, minutes, seconds, direction):
    """
    Convert coordinates in degrees, minutes, seconds to decimal degrees.
    
    Parameters:
    degrees (int): Degree part of the coordinate
    minutes (int): Minutes part of the coordinate
    seconds (float): Seconds part of the coordinate
    direction (str): 'N', 'S', 'E', or 'W'
    
    Returns:
    float: Decimal degrees
    """
    decimal_degrees = degrees + minutes/60 + seconds/3600
    
    # Apply negative sign for South or West
    if direction in ['S', 'W']:
        decimal_degrees = -decimal_degrees
    
    return decimal_degrees


def parse_dms_string(dms_str):
    """
    Parse a DMS string into decimal degrees.
    Accepts formats like:
    - 40° 26' 46" N
    - 40°26'46"N
    - 40 26 46 N
    
    Parameters:
    dms_str (str): String containing DMS coordinates
    
    Returns:
    float: Decimal degrees
    """
    # Remove any special characters and split the string
    dms_str = dms_str.strip()
    
    # Check if the direction is at the end
    direction = None
    if dms_str[-1] in ['N', 'S', 'E', 'W']:
        direction = dms_str[-1]
        dms_str = dms_str[:-1]
    
    # Clean the string and extract numbers
    dms_str = dms_str.replace('°', ' ').replace("'", ' ').replace('"', ' ')
    parts = re.findall(r'[\d.]+', dms_str)
    
    if len(parts) >= 3:
        degrees = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
    elif len(parts) == 2:
        degrees = float(parts[0])
        minutes = float(parts[1])
        seconds = 0
    else:
        degrees = float(parts[0])
        minutes = 0
        seconds = 0
    
    return dms_to_decimal(degrees, minutes, seconds, direction)


def distance_from_latlon(point:tuple, lat_lons:np.ndarray):
    """
    Calculate the distance in km between a point and an array of lat/lon coordinates.
    
    Parameters:
    point (tuple): A tuple containing the latitude and longitude of the point (lat, lon).
        degrees
    lat_lons (np.ndarray): A 2D array of shape (n, 2) where each row is a pair of latitude and longitude coordinates.
        degrees
    The first column should be latitude and the second column should be longitude.
    
    Returns:
    np.ndarray: An array of distances in kilometers from the point to each coordinate in lat_lons.
    """

    # Convert to radians
    origin = np.array([[np.radians(point[0]), np.radians(point[1])]])
    destinations = np.radians(lat_lons)
    # Calculate distances (returns in radians)
    distances_rad = haversine_distances(origin, destinations)
    # Convert to kilometers (Earth radius = 6371 km)
    distances_km = distances_rad * 6371

    # Return
    return distances_km[0]