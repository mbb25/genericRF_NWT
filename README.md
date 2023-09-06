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
Most of the work here is done with jupyter notebooks. Here's an overview of their functionality:

- 1_data_extraction.ipynb
  - Ingests shapefiles and rasters from the raw data, and extracts pixel data from them. This will include R/G/B data, class (and class certainty), and heightmap category. The resulting dataframe is output as a `.csv` file
- 2_process_training_data.ipynb
  - Ingests the `.csv` output from the first notebook, and creates derived columns such as luminance, chrominance, and other features. This notebook has remained largely unchanged from its original form. It outputs a second `.csv` file, with the label `transformed` appended to the filename.
- 2_process_validation_site.ipynb
  - This is only used if you want to run inference on every pixel in a site (for qualitative evaluation, for example). A site name is specified at the top of the notebook, and the script then splits the site into manageable chunks. These chunks are output as pickle files, as this is easier to manage for massive tensors with minimal overhead.
- 3_training_and_inference.ipynb
  - There are custom cross-validation and random search algorithms here. This is to account for the spatial autocorrelation of this data; lichen pixels within one site might look much more similar than between two different sites. To counter this, we do cross-validation by holding out different subsets of the sites.
- 4_composite_clean.ipynb
  - I actually have not touched this one, so I'm not 100% sure if it's still usable



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
