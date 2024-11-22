# -*- coding: utf-8 -*-
"""
Created on Sun May  7 18:10:58 2023

@author: MABEB16
"""

import pandas as pd
import rasterio as rio
import numpy as np
import os
import pickle as pkl
from scipy.ndimage import zoom
import re

def load_site_data(data_dir: str, site: str):
    """
    Retrieve RGB and canopy height data from source files 
    (R,G,B-orthomosaic and canopy-height-model (chm) file) and merge to one raster file.
    Note: The orthomosaic has a finer resolution than the chm file, so the latter is 
    downscaled using the scipy.ndimage 'zoom' function.
    
    Output them as a 4 layer raster (R, G, B, chm).
    Note: R, G, B values are normalized in this function.
    
    Parameters:
    -----------
    data_dir: str
    Directory conatining source files. 
    
    site: str
    Name of site that data is searched for.
    
    Returns:
    --------
    rasters: 6 layer raster
    Raster containing site data (veg-class, R, G, B, chm, certainty)
    
    """
    
    # TODO file paths should not be (partly) hardcoded
    rgb_file = os.path.join(data_dir, site, f'{site}_transparent_mosaic_group1.tif')
    chm_file = os.path.join(data_dir, site, f'dem_and_derivatives/{site}_lp_chm_stratified.tif')
    
    rasters = {}

    # load RGB rasters and metadata
    with rio.open(rgb_file) as f:
        rasters['meta'] = f.meta
        rasters['R'] = f.read((1)).astype(float)/255.0
        rasters['G'] = f.read((2)).astype(float)/255.0
        rasters['B'] = f.read((3)).astype(float)/255.0
        
    # load chm, resize to match RGB rasters
    with rio.open(chm_file) as f:
        height_scale = rasters['meta']['height'] / f.height
        width_scale = rasters['meta']['width'] / f.width
        rasters['chm'] = zoom(f.read((1)).astype(np.int8), (height_scale, width_scale), order=0)

    return rasters


def feature_mission(rasters, n_chunks, run_dir):
    """
    Calculate a list of features necessary to classify the entire orthomosaic. 
    For quicker processing, the orthomosaic is cut into n_chunks and processed
    chunk by chunk.
    
    The feature list is:
    r: normalized red
    g: normalized green
    b: normalized blue
    chm: canopy height
    rc: red chromaticity
    bc: blue chromaticity
    gc: green chromaticity
    rc_d_gc: rc/gc
    rc_p_gc: rc*gc
    ExG: Excess Green Index
    ExR: Excess Blue Index
    ExB: Excess Red Index
    ExGmExR: 
    Ikaw:
    MGRVI:
    GLI:
    Y:
    L:
    z_score_Y:
    z_score_L:
    
    Parameters:
    -----------
    rasters: raster
    5 layer raster with metadata, r,g,b and chm data. Output from 'load_data' function.
    
    n_chunks: int
    The original input raster is going to be cut into n x n chunks for faster processing.
    
    run_dir: str
    Directory into which the processed chunks are saved (23 layer rasters saved as .pkl file).
    
    Returns:
    --------
    None, saves 23 layer rasters as .pkl files.
    
    """
    eps = 1e-9
    
    chunk_height = rasters['meta']['height'] / n_chunks
    chunk_width = rasters['meta']['width'] / n_chunks
    
    width_track = []
    
    # Features - to save memory, I'm allocating the array in advance
    # and using pseudo enumerators to index the layers
    R = 0
    G = 1
    B = 2
    chm = 3
    rc = 4
    gc = 5
    bc = 6
    rc_d_gc = 7
    rc_p_gc = 8
    ExG = 9
    ExR = 10
    ExB = 11
    ExGmExR = 12
    Ikaw = 13
    MGRVI = 14
    GLI = 15
    Y = 16
    L = 17
    z_score_Y = 18
    z_score_L = 19

    for y in range(n_chunks):
        for x in range(n_chunks):
            y_min = int(y * chunk_height)
            y_max = int(min((y+1) * chunk_height, rasters['meta']['height']))

            x_min = int(x * chunk_width)
            x_max = int(min((x+1) * chunk_width, rasters['meta']['width']))

            # 20 features total
            chunk = np.zeros((20, y_max-y_min, x_max-x_min))
            width_track.append((f"chunk_{y}_{x}", chunk.shape[2]))

            chunk[R] = rasters['R'][y_min:y_max,x_min:x_max]
            chunk[G] = rasters['G'][y_min:y_max,x_min:x_max]
            chunk[B] = rasters['B'][y_min:y_max,x_min:x_max]
            chunk[chm] = rasters['chm'][y_min:y_max,x_min:x_max]

            # shortcircuit if there are no valid pixels
            check = np.where(chunk[chm] >= 0)
            if len(check[0]) == 0:
                print(f'Skipping chunk {y}-{x}')
                continue

            chunk[rc] = chunk[R] / (chunk[R] + chunk[G] + chunk[B] + eps)
            chunk[gc] = chunk[G] / (chunk[R] + chunk[G] + chunk[B] + eps)


            # --------------------------------------------------------------------------------------
            # extending pada data frame by rc/gc, rc+gc, luminance Y, perceived lightness L, z_score_Y, z_score_L 
            # Excess indecies ExG, ExR, ExB, indices ExGmExR, Ikaw, MGRVI, and GLI 
            # --------------------------------------------------------------------------------------
            # blue chromaticity
            chunk[bc] = 1-(chunk[rc]+chunk[gc])

            # ration rc/gc
            chunk[rc_d_gc] = chunk[rc]/(chunk[gc] + eps)

            # sum rc+gc
            chunk[rc_p_gc] = chunk[rc]+chunk[gc]

            # excess indecies
            chunk[ExG] = 2*chunk[G] - chunk[R] - chunk[B]

            chunk[ExR] = 1.4*chunk[R] - chunk[G]

            chunk[ExB] = 1.4*chunk[B] - chunk[G]

            # index ExGmExR
            chunk[ExGmExR] = chunk[ExG]-chunk[ExR]

            # index Ikaw - Kawashima Index
            chunk[Ikaw] = (chunk[R]-chunk[B])/(chunk[R]+chunk[B]+eps)

            # MGRVI - modified green red vegetation index
            chunk[MGRVI] = ((chunk[G])**2 - (chunk[R])**2)/((chunk[G])**2 + (chunk[R])**2 + eps)

            # GLI - green leaf index
            chunk[GLI] = (2*chunk[G] - chunk[R] - chunk[B])/(2*chunk[G] + chunk[R] + chunk[B] + eps)

            # luminance Y
            sRGBtoLin = lambda x: np.where(x <= 0.04045, x/12.92, (pow(((x + 0.055)/1.055), 2.4)))
            chunk[Y] = 0.2126*sRGBtoLin(chunk[R]) + 0.7152*sRGBtoLin(chunk[G]) + 0.0722*sRGBtoLin(chunk[B])

            # perceived lightness L
            YtoLstar = lambda y: np.where(y <= (216/24389), y*(24389/27), (pow(y, (1/3)) * 116 - 16))
            chunk[L] = YtoLstar(chunk[Y])

            # z_scores for Y and L
            valid_Y_np = chunk[Y][check]
            Y_mean = valid_Y_np.mean()
            Y_std = valid_Y_np.std()
            del valid_Y_np

            valid_L_np = chunk[L][check]
            L_mean = valid_L_np.mean()
            L_std = valid_L_np.std()
            del valid_L_np

            chunk[z_score_Y] = (chunk[Y] - Y_mean)/(Y_std+eps)
            chunk[z_score_L] = (chunk[L] - L_mean)/(L_std+eps)
            file_path = os.path.join(run_dir, 'composite', f'chunk_{y}_{x}.pkl')
            print(f'Writing chunk {y}-{x}')
            with open(file_path, "wb") as file:
                # Use the pickle.dump() function to write the dictionary to the file
                pkl.dump(chunk, file)
    
    with open(run_dir+'/composite/width_track.txt', 'w') as outfile:
        outfile.write('\n'.join(re.sub("['(),]",'', str(i)) for i in width_track))
    outfile.close()            


def classify_mission(rgb_file, run_dir, model, blocks=4):
    """
    This function uses a trained model to classify an entire orthomosaic. 
    The classified orthomosaic is written out as tif using the metadata from
    the original's orthomosaic (rgb_file).
    
    
    Parameters:
    -----------
    rgb_file: str
    directory and name of orthomosaic from which meta data for new, classified
    file is to be copied.
    
    run_dir: str
    Directory containing chunks with the necessary features to classify the 
    orthomosaic.
    
    model:
    trained model object, saved as .pkl file.
    
    blocks: int, default 4
    Indicates into how many chunks the input orthomosaic has been split.  
    
    Returns:
    --------
    Output: np array
    An array with the size of the original orthomosaic containing one class per pixel.
    
    """
    #TODO chm position hard coded
    chm=3
    with rio.open(rgb_file) as f:
        meta = f.meta
    
    curr_y = 0
    curr_x = 0
    h = 0
    w = 0
    
    output = np.zeros((meta['height'], meta['width']))
    
    width_track = {}
    with open(f"{run_dir}/composite/width_track.txt") as f:
        for line in f:
            (key, val) = line.split()
            width_track[key] = int(val)
    
    for y in range(blocks):
        for x in range(blocks):
            print(f'Processing block {y}-{x}')
            
            file_path = os.path.join(run_dir, 'composite', f'chunk_{y}_{x}.pkl')
                    
            if not os.path.isfile(file_path):
                w = width_track[f'chunk_{y}_{x}']
                curr_x += w
                continue
                
            with open(file_path, "rb") as file:
                # Use pickle.load() to read the dictionary from the file
                block = pkl.load(file)
            
            h = block.shape[1]
            w = block.shape[2]
            
            block_output = np.zeros((h, w))
            
            # get the coordinates of the relevant pixels
            py, px = np.where(block[chm] >= 0)
            
            if len(py) > 0:
                features = block[:, py, px].T
                block_output[py, px] = model.predict(features)
                output[curr_y:curr_y+h, curr_x:curr_x+w] = block_output
                
                del features
        
            curr_x += w
            del block, block_output, py, px

        curr_x = 0
        curr_y += h
        
    return output


def inpsectiontocsv(savepath, savename, clf, cf_matrix, training_data, testing_data, selection=[]):
    with open(savepath+savename, "w", newline='') as f:
        print(savename.split('.')[0], file=f)
        print('selection,%s' % str(selection)[1:-1].replace("'", ""), file=f)

        X_res = training_data[0]
        y_res = training_data[1]
        X_test = testing_data[0]
        y_test = testing_data[1]

        print ('training_score,%s' % round(clf.score(X_res, y_res), 4), file=f)
        print('testing_score,%s' % round(clf.score(X_test, y_test), 4), file=f)

        print('confusion_matrix:', file=f)
        labels=clf.classes_
        cf_df = pd.DataFrame(cf_matrix, index=labels, columns=labels)
        cf_df.to_csv(f, sep=',', index=True, header=True)

        feature_importance = pd.DataFrame(clf.feature_importances_, index=X_res.columns).sort_values(by=0, ascending=False)
        print('feature_importance:', file=f)
        feature_importance.to_csv(f, sep=',', header=False)
