# General vegetation classifier for drone data collected in NWT

## Project description
Changing illumination conditions (eg. clear sky vs. overcast sky) introduce large variations in RGB imagery collected by drones. As a consequence, vegetation like trees, mosses or lichen 
may appear differently in drone missions flown under different illumination conditions. This variation makes it challenging to use one classifier to classify vegetation in drone imagery obtained 
from different missions. One solution to still obtain a robust classification is to create mission-specific classifiers. However, creating a robust classifier is laborious as it requires a large volume of 
training data, usually gained through manual labelling of the image. Having mission-specifc classifiers which each require a new collection of training data is thus not an ideal solution. 
The objective of this project is to generate a vegetation classifier that performs well accross missions despite changing illumination conditions during image acquisition.

The features that are currently used for the general classifier are all derivatives of RGB and canopy height data, both of which are easily obtained from standard drone imagery. The final list of features is 
not available at this moment, as the developement of the classifier is still work in progress. A complete list of features will be added to this documentation after the development of the classifier 
has been completed.

The code in this repository consists of several notebooks as well as auxiliary python files that contain functions which are used in the notebooks. Notebooks 1-3 deal with the creation and 
processing of training data as well as the training and evaluation of the random forest classifier. Notebook 4 uses trained classifiers to classify an entire mission area. **Please note that Notebook 4
is currently only used for development purposes and should not be used otherwise before the developement of the classifier is completed**


## How to use the classifier
*work in progress* 


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
