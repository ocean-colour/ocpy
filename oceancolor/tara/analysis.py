""" Tara related analysis """

from shapely.geometry import Point

import geopandas as gpd
from geopandas import GeoDataFrame as gdf
import pandas as pd


def dist_coast():
    # pull in high res coastline datasets from https://www.naturalearthdata.com/downloads/10m-physical-vectors/
    # get the coastline and minor islands at 10m
    coastlines = gpd.read_file('data/ne_10m_coastline.shp')
    islands = gpd.read_file('data/ne_10m_minor_islands_coastline.shp')

    coastlines = pd.concat([coastlines, islands])

    def min_distance(point, lines):
        return lines.distance(point).min()

    coastline =coastlines.to_crs('EPSG:3087')

    # gdf is the main tara geodataframe
    gdf = gdf.to_crs('EPSG:3087') # must be a projection with meters as the unit to calculate distance in meters accurately such as https://epsg.io/3087

    gdf['min_dist_to_coast'] = gdf.geometry.apply(min_distance, args=(coastline,))

    gdf = gdf.to_crs('EPSG:4326') # then convert back to WGS84 lat lon projection