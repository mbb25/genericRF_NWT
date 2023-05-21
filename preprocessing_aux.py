# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 11:45:47 2023

@author: User
"""
import numpy as np
# import pandas as pd

def excess_index(R, G, B, excessband='green', convert_to_255=False):
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
    mean = x_all.mean()
    sd = x_all.std()
    z = []
    for val in x_all:
        z_val = (val - mean)/sd
        z.append(z_val)

    return(z)

def brightness(r, g, b):
    brightness = []
    for i, val in enumerate(r):
        #vR = val/255
        #vG = g[i]/255
        #vB = b[i]/255

        brightness_val = (r+g+b)/3
        brightness.append(brightness_val)

    return(brightness)


def sRGBtoLin(colorChannel, single_value=True):
        # Send this function a decimal sRGB gamma encoded color value
        # between 0.0 and 1.0, and it returns a linearized value.

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
        #Send this function a luminance value between 0.0 and 1.0,
        #and it returns L* which is "perceptual lightness"
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

# function to calculate Mahalanobis distance
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
