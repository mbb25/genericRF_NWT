# General vegetation classifier for drone data collected in NWT

## Project description
Changing illumination conditions during data aquisition (eg. clear sky vs. overcast sky) introduce large variations in RGB imagery and modify the appearance of vegetation like trees, mosses or lichen 
in drone data. These variations in RGB imagery complicate using one generic classifier to classify vegetation across different drone datasets. Currently, robust classification is obtained by mission-specific classifiers, however, 
training new classifers for each new drone dataset is laborious and not ideal. The objective of this project is to generate a vegetation classifier that performs well accross missions despite changing illumination conditions 
during image acquisition.

The features that are currently used for training the classifiers are all derivatives of RGB and canopy height data, both of which are easily obtained from standard drone imagery. The final list of features is 
not available at this moment, as the developement of the classifier is still work in progress. A complete list of features will be added to this documentation after the development of the classifier 
has been completed.

This repository has two sections. The first, train_classifier, contains code to train site-specific random forest (RF) classifiers as well as code to experiment with generic RF classifiers. This section is for development. 
The second, use_classifier, contains notebooks and code for classifying RGB drone datasets into a list of vegetation types (see below). Multiple generic as well as site-specific classifiers are stored in the classifier_lib and 
available for classification. This section is for users.

## Vegetation types
* 1 - cladonia lichen
* 2 - crustous lichen, dark
* 3 - crustous lichen, light
* 4 - rock
* 5 - broadleaf
* 6 - needleleaf
* 7 - deadwood (ground)
* 8 - graminoids, yellow
* 9 - moss
* 10 - soil
* 11 - water
* 12 - low-green
* 13 - dry-branches (in canopy)
* 14 - graminoids, green


## How to use the classifier
wip

## Publication
*in preparation*

## Code authors
Maria Belke-Brea (mbelkebrea@wlu.ca),
Matthew Fortier,
Owen Lucas,
Martin Weiss

## Project collaborators
Maria Belke-Brea (Wilfrid Laurier University),
Jurjen van der Sluijs (Government of the Northwest Territories),
Robert Fraser (Natural Resources Canada),
Owen Lucas (Natural Resources Canada),
Matthew Fortier (MILA Quebec Artifical Intelligence Institute),
Martin Weiss (MILA Quebec Artificial Intelligence Institute),
Jennifer Baltzer (Wilfrid Laurier University),
Steve Cumming (Université Laval)
