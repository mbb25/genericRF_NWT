# -*- coding: utf-8 -*-
"""
Created on Wed July 25 13:20:48 2024

@author: Matthew Fortier, Maria Belke Brea
This module is used for creating a general lichen classifier for highres UAV data (0.5 - 5 cm) 
collected in NWT, Canada.
The functions in this module retrieve RGB and chm information for labelled pixels and store it in 
a csv file. 
The csv file is then the starting point for training a general RF classifier for cladonia lichen.

"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio.features import rasterize

def process_overlapping_shapes(label_gdf, chm_gdf):
    """ Takes label and chm geometries, and returns a combined gdf where each
    label is associated with a single canopy height.
    
    Parameters
    ----------
    label_gdf: geopandas dataframe
    A dataframe listing all label polygones together with their geolocation
    
    chm_gdf: geopandas dataframe
    A dataframe listing all canopy height polygones togehter with their geolocation
    
    Returns
    -------
    combined_gdf: geopandas dataframe
    
    """
    
    combined_gdf = label_gdf.copy()
    combined_gdf['DN'] = -1
    chm_geometry = chm_gdf.geometry
    
    # iterate through each label shape
    for index, row in label_gdf.iterrows():
        intersecting_bools  = chm_geometry.intersects(row.geometry)
        intersecting_shapes = intersecting_bools[intersecting_bools != False]
        chm_values = list(chm_gdf.iloc[intersecting_shapes.index]['DN'])
        if len(chm_values) > 0:
            if row['class'] in [5, 6, 13]: # ['broadleaf', 'needleleaf', 'dry_branches']
                label_chm = max(chm_values)
            elif row['class'] in [1, 4, 7, 8, 9, 10, 12]: # ['lichen', 'rock', 'deadwood', 'graminoids', 'moss', 'soil', 'low_green']
                label_chm = min(chm_values)
            combined_gdf.loc[index, 'DN'] = label_chm
        else:
            combined_gdf.drop([index], inplace=True)
    return combined_gdf


def load_site_data(data_dir: str, site: str, include_targeted_labels=False):
    """ Retrieves class, RGB, canopy height, and certainty data from source files.
    Outputs them as a 6 layer raster (class, R, G, B, chm, certainty).
    
    Parameters
    ----------
    site: str
    The name of the site for which data is retrieved.
    
    include_targeted_labels: bool 
    Allows to retrieve data from an additional file that stores data from targeted labelling (in contrast to random labelling).
    (Default is False).
    
    Returns:
    --------
    output_raster: raster
    6 layer raster containing data on class, R, G, B, chm and class certainty.
    """ 
    
    # TODO: remove hardcoded datapaths
    rgb_file = os.path.join(data_dir, site, f'{site}_hp_transparent_mosaic_group1.tif')
    chm_file = os.path.join(data_dir, site, 'dem_and_derivatives', f'{site}_hp_chm_stratified.tif')
    chm_shp_file = os.path.join(data_dir, site, 'stratified_sampling', f'{site}_hp_chm_stratified.shp')
    label_file = os.path.join(data_dir, site, f'{site}_hp_labelledpoints.gpkg')

    # load RGB raster and metadata
    with rio.open(rgb_file) as f:
        # Get raster metadata
        height = f.height
        width = f.width
        crs = f.crs
        transform = f.transform
        output_raster = np.ndarray((6,height,width), dtype=np.uint8)
        rgb_raster = f.read((1,2,3), out=output_raster[1:4,:,:])
        
    # load shapefiles
    if include_targeted_labels == True:
        label_gdf_random = gpd.read_file(label_file).to_crs(crs).dropna(subset=['class', 'certainty'])
        label_gdf_targeted = gpd.read_file(label_file, layer=f'{site}_hp_targeted_labelling').to_crs(crs).dropna(subset=['class', 'certainty'])
        label_gdf = pd.concat([label_gdf_random, label_gdf_targeted], ignore_index=True)
    else:
        label_gdf = gpd.read_file(label_file).to_crs(crs).dropna(subset=['class', 'certainty'])
    label_gdf['certainty'] = label_gdf['certainty'].astype(np.uint8)
    chm_gdf = gpd.read_file(chm_shp_file).to_crs(crs)
    
    combined_gdf = process_overlapping_shapes(label_gdf, chm_gdf)
    
    rasterize(
        ((r['geometry'], r['class']) for _, r in combined_gdf.iterrows()),
        out_shape=(height, width),
        out=output_raster[0,:,:],
        transform=transform
    )
    
    rasterize(
        ((r['geometry'], r['DN']) for _, r in combined_gdf.iterrows()),
        out_shape=(height, width),
        out=output_raster[4,:,:],
        transform=transform
    )
    
    rasterize(
        ((r['geometry'], r['certainty']) for _, r in combined_gdf.iterrows()),
        out_shape=(height, width),
        out=output_raster[5,:,:],
        transform=transform
    )
    return output_raster


def convert_to_dataframe(combined_raster, site, class_map):
    """Produces a list of indices where we have labelled pixels and retrieves site, class, rgb, chm and certainty information.
    
    Parameters:
    -----------
    combined_raster: raster
    6 layer raster containing data on class, R, G, B, chm and class certainty. 
    Combined raster is obtained as output from the load_site_data() function.
    
    site: str
    The name of the site for which data is retrieved.
    
    Returns:
    --------
    df: geopanda dataframe
    Dataframe of class pixels holding info on site, class label, R, G, B, chm, and certainty per pixel.
    
    indices: numpy array
    Indices where we have a label [ [348, 1451], [581, 2099], ... ].
    """
    
    
    indices = np.argwhere(np.isin(combined_raster[0], list(class_map.values())))
    
    labels_array = combined_raster[0, indices[:,0], indices[:,1]].reshape(-1, 1)
    labels_df = pd.DataFrame(labels_array, columns=['veg_class'])
    
    rgb_features = combined_raster[1:4, indices[:,0], indices[:,1]].T
    rgb_df = pd.DataFrame(rgb_features, columns=['R', 'G', 'B'])
    
    chm_features = combined_raster[4, indices[:,0], indices[:,1]].reshape(-1, 1)
    chm_df = pd.DataFrame(chm_features, columns=['chm'])
    
    certainty_array = combined_raster[5, indices[:,0], indices[:,1]].reshape(-1, 1)
    certainty_df = pd.DataFrame(certainty_array, columns=['class_certainty'])
    
    site_df = pd.DataFrame({'site': [site] * len(labels_df), 'y_pos': indices[:,0], 'x_pos': indices[:,1]})
    
    df = pd.concat([site_df, labels_df, rgb_df, chm_df, certainty_df], axis=1)
    return df, indices

def normalize_rgb(df):
    """
    Normalizes RGB values to range from 0 to 1 (instead of 0-255).
    
    Parameters:
    -----------
    df: dataframe
    Dataframe that has to contain 3 columns each with R, G, and B as column name.
    
    Returns:
    --------
    df: dataframe
    Dataframe with normalized RGB values.
    
    """
    for c in ['R', 'G', 'B']:
        df[c] = df[c].astype(float)/255.0
    
    return df