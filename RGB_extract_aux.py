# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 17:10:48 2023

@author: MABEB16
"""
from rasterio.mask import mask
import itertools

def label_filter(gdf, classnames=[], classvals=[], colname2filter='class'):
    gdf_filtered = {}
    for i, name in enumerate(classnames):
        val = classvals[i]
        gdf_filtered[name] = gdf[gdf[colname2filter]==val]
    return(gdf_filtered)


def extractDataPerLabel(raster, filtergdf, lat_array, lon_array, classnames=[],
                        red_index=0, green_index=1, blue_index=2):

    RGB = {}
    LatLon = {}
    RcGc = {}
    for name in classnames:
        sub_dataset = filtergdf[name]
        if len(sub_dataset) > 0:
            shapes = [feature for feature in sub_dataset['geometry']]

            # use shape list in mask function, set crop to False to maintain ortho dimensions
            out_image, out_transform = mask(raster, shapes, crop=False)

            # get RGB for lichen pix
            out_red = out_image[red_index][out_image[red_index]>0]
            out_green  = out_image[green_index][out_image[green_index]>0]
            out_blue  = out_image[blue_index][out_image[blue_index]>0]

            RGB[name] = {}
            RGB[name]['red'] = out_red
            RGB[name]['green']  = out_green
            RGB[name]['blue']  = out_blue

            # get lons and lats for lichen pix
            LatLon[name] = {}
            LatLon[name]['lon'] = lon_array[out_image[0] != 0]
            LatLon[name]['lat'] = lat_array[out_image[0] != 0]

            # get Rc, Gc
            rc = []
            gc = []
            for i, val in enumerate(out_red):
                rc.append(int(val)/(int(val)+int(out_green[i])+int(out_blue[i])))
                gc.append(int(out_green[i])/(int(val)+int(out_green[i])+int(out_blue[i])))

            RcGc[name] = {}
            RcGc[name]['rc'] = rc
            RcGc[name]['gc'] = gc
        else:
            RGB[name] = {}
            RGB[name]['red'] = []
            RGB[name]['green'] = []
            RGB[name]['blue'] = []

            LatLon[name] = {}
            LatLon[name]['lon'] = []
            LatLon[name]['lat'] = []

            RcGc[name] = {}
            RcGc[name]['rc'] = []
            RcGc[name]['gc'] = []

    return(RGB, LatLon, RcGc)

def extractCanopyHeightPerLabel(raster, chm_gdf, filtergdf, classnames):

    # get chm polygons
    chm_geometry = chm_gdf.geometry

    chm_strata = {}

    # extract chm data for each class
    for name in classnames:
        # filtergdf is a function that filters for the chosen veg_class given in names
        sub_dataset = filtergdf[name]

        # only proceed if the veg_class is found in the gdf, otherwise give empty output
        # (some veg types may not be found in specific orthos)
        if len(sub_dataset) > 0:
            #s(name)

            # get veg_class polygons and then loop through them for intersect analysis
            label_shapes = [feature for feature in sub_dataset['geometry']]
            chm_strata_tmp = []
            for label_shape in label_shapes:
                # --------------------------------------------------------------------------------
                # use orthoraster to know how many pixels the veg_class polygone covers
                # the final output list will have one row for each RGB value per pixel
                # in the veg_class polygone. So for the chm and RGB data to be of same
                # length, chm data needs to be multiplied by the pixel number.
                # !!! this is a HORRIBLE and VERY time consuming way of doing this,
                # !!! needs another solution
                out_image, out_transform = mask(raster, [label_shape], crop=False)
                # all pixels in raster that are outside of the veg_class pixel are 0
                # out_image[0] is the red channel, choses randomly, could also be green or blue.
                pixel_num = len(out_image[0][out_image[0]>0])

                # --------------------------------------------------------------------------------
                # Intersection analysis, which chm polygone does the veg_class polygone intersect with?
                # isIntersection is a list of length=chm polygones. Each polygone that does not intersect
                # with the veg_class polygone has entry = False, the ones that intersect = True.
                isIntersection=chm_geometry.intersects(label_shape)
                new_isInt = isIntersection[isIntersection != False]

                # --------------------------------------------------------------------------------
                # In rare cases, new_isInt can be a list instead of a single value when veg_class
                # polygone intersects with several chm polygons.
                # But result should be only one chm class, so a value is chosen depending on veg_class
                # 'DN' is the chm class, the output we are looking for. It can be
                # 1 = low, 2 = intermediate, or 3 = high
                if len(new_isInt) == 1:
                    subdata_chm = int(chm_gdf.iloc[new_isInt.index]['DN'])
                elif len(new_isInt) > 1:
                    subdata_chm = chm_gdf.iloc[new_isInt.index]['DN']
                    if name in ['broadleaf', 'needleleaf', 'dry_branches']:
                        subdata_chm = int(max(subdata_chm))
                        # because tree pixels are more likely in the higher strata
                    elif name in ['lichen', 'rock', 'deadwood', 'graminoids', 'moss', 'soil', 'low_green']:
                        subdata_chm = int(min(subdata_chm))
                        # because lichen are growing low on the ground.
                else: # if new_isInt is empty because the veg_class polygon overlays with a hole in the chm
                    subdata_chm=-1

                # multiply with pixelnum to assure same length with RGB list
                subdata_chm = [subdata_chm]*pixel_num
                chm_strata_tmp.append(subdata_chm)
        else:
            chm_strata_tmp=[]

        # unnest chm_strata_tmp list
        chm_strata_flat = list(itertools.chain.from_iterable(chm_strata_tmp))
        chm_strata[name] = chm_strata_flat

    return(chm_strata)