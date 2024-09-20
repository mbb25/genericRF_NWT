# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 11:45:47 2023

@author: Matthew Fortier, Maria Belke Brea
This module is used for creating a general lichen classifier for highres UAV data (0.5 - 5 cm) 
collected in NWT, Canada.
The functions in this module calculate a list of features (see help add_features) based on RGB and chm data.
The purpose of these features is to improve the classifier training.
"""
import numpy as np
# import pandas as pd

def add_features(df, feature_list=[], all_features=False):
    """Function that calculates a list of features based on RGB data. 
    When used as training data for a RF classifier, these features can improve classifier performance.
    The user can either choose to calculate all features by activating the all_features argument or only a subset of features
    specified in the feature_list argument (see below). The new features are added to the given dataframe df.
    
    Parameters:
    -----------
    df: dataframe
    Dataframe required to have the following columns: site, R, G, and B. 
    
    feature_list: list (default: empty list)
    List containing features to calculate. Feature names are to be given as str and can be:
    - 'chromaticity'
    - 'rc/gc'
    - 'rc+gc'
    - 'excess indecies'
    - 'Ikaw' (index Ikaw - Kawashima Index)
    - 'MGRVI' (modified green red vegetation index)
    - 'GLI' (green leaf index)
    - 'brightness'
    
    all_features: bool (default:False)
    Set to True when all features are to be calculated, instead of using the feature_list argument.
    
    Returns:
    --------
    df: dataframe
    Outputs the dataframe given in the df argument extended by the selected features in features list 
    or all features if all_features arguemtn is active. Each feature added as new column.
    """
    
    if 'chromaticity' in feature_list or all_features == True:
        df['rc'] = df['R'] / (df['R'] + df['G'] + df['B'])
        df['gc'] = df['G'] / (df['R'] + df['G'] + df['B'])
        df['bc'] = 1-(df['rc']+df['gc'])
        
    if 'rc/gc' in feature_list or all_features == True:
        df['rc/gc'] = df['rc']/df['gc']
        
    if 'rc+gc' in feature_list or all_features == True:
        df['rc+gc'] = df['rc']+df['gc']
        
    if 'excess indecies' in feature_list or all_features == True:
        ExG = [excess_index(G=val, R=np.array(df['R'])[i], B=np.array(df['B'])[i], convert_to_255=False)
               for i, val in enumerate(np.array(df['G']))]
        df['ExG'] = ExG

        ExR = [excess_index(G=val, R=np.array(df['R'])[i], B=np.array(df['B'])[i], convert_to_255=False, excessband='red')
               for i, val in enumerate(np.array(df['G']))]
        df['ExR'] = ExR

        ExB = [excess_index(G=val, R=np.array(df['R'])[i], B=np.array(df['B'])[i], convert_to_255=False, excessband='blue')
               for i, val in enumerate(np.array(df['G']))]
        df['ExB'] = ExB
        
        df['ExGmExR'] = df['ExG']-df['ExR']
        
        
    if 'Ikaw' in feature_list or all_features == True:
        df['Ikaw'] = (df['R']-df['B'])/(df['R']+df['B'])
        
    if 'MGRVI' in feature_list or all_features == True:
        df['MGRVI'] = ((df['G'])**2 - (df['R'])**2)/((df['G'])**2 + (df['R'])**2)
        
    if 'GLI' in feature_list or all_features == True:
        df['GLI'] = (2*df['G'] - df['R'] - df['B'])/(2*df['G'] + df['R'] + df['B'])
        
    if 'brightness' in feature_list or all_features == True:
        # luminance Y
        Y = [0.2126*sRGBtoLin(val) + 
             0.7152*sRGBtoLin(np.array(df['G'])[i]) + 
             0.0722*sRGBtoLin(np.array(df['B'])[i]) for i, val in enumerate(np.array(df['R']))]
        df['Y'] = Y
        
        # perceived lightness L
        L = [YtoLstar(val) for val in df['Y']]
        df['L'] = L
        
        # z_score of the luminance Y
        df['z_score_Y'] = 0
        sites = set(df['site'])
        for site in sites:
            condition = df['site'] == site
            subdata_site = df[df['site'].isin([site])]['Y']
            z_site = z_score(x_all=np.array(subdata_site))
            df.loc[condition, 'z_score_Y'] = z_site
        
        # z_score of the perceived lightness L
        df['z_score_L'] = 0
        sites = set(df['site'])
        for site in sites:
            condition = df['site'] == site
            subdata_site = df[df['site'].isin([site])]['L']
            z_site = z_score(x_all=np.array(subdata_site))
            df.loc[condition, 'z_score_L'] = z_site
    
    return df

def excess_index(R, G, B, excessband='green', convert_to_255=False):
    """
    Use R (G or B) values to calculate a red (green or blue) excess index. 
    
    Parameters:
    -----------
    R, G, B: float
    Normalized Red, Green and Blue values. 
    
    excessband: str
    Specify which excess index should be calculated. Possible are 'green', 'red' or 'blue'.
    
    convert_to_255: bool
    Set to 'True' to transfer the normalized R, G, B values into the 255 color chart.
    
    Returns:
    --------
    Ex_ind: int or float
    Green, Red or Blue excess index as specified in 'excessband' parameter.
    
    """
    if convert_to_255:
        R = R*255
        G = G*255
        B = B*255

    if excessband == 'green':
        Ex_ind = 2*G - R - B
    elif excessband == 'red':
        Ex_ind = 1.4*R - G
    elif excessband == 'blue':
        Ex_ind = 1.4*B - G
    else:
        print('excessband must be either "green", "red" or "blue"')

    if convert_to_255:
        return(int(Ex_ind))
    else:
        return(Ex_ind)

def z_score(x_all):
    """
    Calculates a z_score for every given value in a dataset.
    
    Parameters:
    -----------
    x_all: np.array or list
    Dataset for which z-scores should be calculated.
    
    Returns:
    --------
    z: list
    List containing one z-score per given value in x_all.
    
    """
    mean = x_all.mean()
    sd = x_all.std()
    z = []
    for val in x_all:
        z_val = (val - mean)/sd
        z.append(z_val)

    return(z)

def brightness(r, g, b):
    """
    Calculates pixel brightness based on given color triples (rgb). 
    
    Parameters:
    -----------
    r, g, b: np.array or list
    Color values to calculate brightness.
    
    Returns:
    --------
    brightness: list
    List containing one brightness value per given color triple (rgb).
    
    """
    brightness = []
    for i, val in enumerate(r):
        #vR = val/255
        #vG = g[i]/255
        #vB = b[i]/255

        brightness_val = (r+g[i]+b[i])/3
        brightness.append(brightness_val)

    return(brightness)


def sRGBtoLin(colorChannel, single_value=True):
    """
    Send this function a decimal sRGB gamma encoded color value
    between 0.0 and 1.0, and it returns a linearized value. 
    This is a helper function to calculate luminance values (Y).
    
    Parameters:
    -----------
    colorChannel: float or list
    r, g, or b gamma encoded (normalized) color value. 
    
    #TODO test for float or list inside the funtion and delete this parameter.
    single_value: bool
    Indicates if colorChannel is a float or a list
    
    Returns:
    --------
    colorChannel: float
    Linearized RGB value
    
    toLinList: list
    List of linearized RGB values
    
    """
    #vColor = colorChannel/255
    if single_value == True:
        if colorChannel <= 0.04045:
            return(colorChannel/12.92)
        else:
            return(pow(((colorChannel + 0.055)/1.055), 2.4))
    else:
        toLinList = []
        for val in colorChannel:
            if val <= 0.04045:
                toLinList.append(val/12.92)
            else:
                toLinList.append(pow(((val + 0.055)/1.055), 2.4))

        return(toLinList)



def YtoLstar(Y, single_value=True):
    """
    Send this function a luminance value (Y) between 0.0 and 1.0,
    and it returns L* which is "perceptual lightness"
    
    Parameters:
    -----------
    Y: float or list
    Luminance value or list, between 0.0 and 1.0
    
    #TODO test for float or list inside the funtion and delete this parameter.
    single_value: bool
    Indicates if Y is a float or a list
    
    Returns:
    --------
    L: float
    Perceptual brightness value
    
    L_list: list
    List of perceptual brightness values
    
    """
    if single_value == True:
        if Y <= (216/24389):      #The CIE standard states 0.008856 but 216/24389 is the intent for 0.008856451679036
            return(Y*(24389/27))  #The CIE standard states 903.3, but 24389/27 is the intent, making 903.296296296296296
        else:
            return(pow(Y, (1/3)) * 116 - 16)
    else:
        L_list = []
        for val in Y:
            if val <= (216/24389):      #The CIE standard states 0.008856 but 216/24389 is the intent for 0.008856451679036
                L_list.append(val*(24389/27))  #The CIE standard states 903.3, but 24389/27 is the intent, making 903.296296296296296
            else:
                L_list.append(pow(val, (1/3)) * 116 - 16)
        return(L_list)

# ------------------------------------------------------------------------------------
# functions to calculate Mahalanobis distance
# TODO Change this two functions to a different .py
def mahalanobis(x=None, data=None, cov=None):

    x_mu = x - np.mean(data)
    print(x_mu)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = np.linalg.inv(cov)
    left = np.dot(x_mu, inv_covmat)
    mahal = np.dot(left, x_mu.T)
    return mahal.diagonal()


def malaha_fun(dataf, veg_type, features):

    # data cloud
    feature1 = features[0]
    feature2 = features[1]
    subdata_class = dataf[dataf['veg_class'].isin([veg_type])][[feature1, feature2]]
    data = np.array(subdata_class)

    # covarince matrix for the data cloud
    data = np.transpose(data)
    covM = np.cov(data, bias=False)
    invCovM = np.linalg.inv(covM)

    # mean vector of data cloud
    m = np.mean(data, axis=1)

    malaha_dist = []
    for i in range(0, subdata_class.shape[0]):
        x = np.array([subdata_class.iloc[i]])
        xMn = x-m

        # malahanobis distance, calculated in three steps
        temp1 = np.dot(xMn, invCovM)
        temp2 = np.dot(temp1, np.transpose(xMn))
        MD = np.sqrt(temp2)

        malaha_dist.append(MD[0][0])

    return(malaha_dist)
