# Extract features and train classifier
The subfolder train_classifier contains all code and notebooks to train an insite random forest classifier as well as to create a generic 
classifier based on training data from multiple missions. 
This folder is for development purposes only.

There are two parts to the code.

## Part I
Extracting the features from a RGB geotif file and a chm shapefile using the label geodatabase. This work is done in the features notebook, auxiliary functions are stored in data_extraction.py and preprocessing_aux.py. The extracted features are first stored in a pandas dataframe, then saved in a csv file.

Files for Part I:
- features.ipynb
- data_extraction.py
- preprocessing_aux.py

Input files:
- Drone_orthomosaic_RGB.tif
- chm_stratified.shp
- manual_labels.gpkg

Output file:
- features_used_for_training.csv

## Part II
The features are now used to train a random forest classifier. Several training approaches are possible. The in_site approach trains one classifier per site (train_insite_RF.ipynb) which has a better performance in most cases but requires having a training data set (label geodatabase) for each site. The generic approach uses features from all sites to train one generic classifier for all sites and which, ideally, will then be generalizable to other drone data collected in southern NWT (train_experiments.ipynb). Several experiments were run to train a generic classifier using different data subsets, features, or data processing steps with the aim to improve the overall performance of the final generic classifier. The auxiliary functions are stored in training.py.

Files for Part II:
- train_insite_RF.ipynb
- train_experiments.ipynb
- training.py

Input file:
- features_used_for_training.csv

Output file:
- trained_RF_classifier.pkl
- classifier_performance.pkl

Note: both the insite and generic classifiers are stored as pkl files in the classifier library in this github repo (classifier_lib). To see how to use the classifiers to classify a new drone mission see notebooks and documentation in the folder use_classifier, which also lives in the repo. 
