# Extract features and train classifier
The subfolder train_classifier contains all code and notebooks to train an insite random forest classifier as well as to create a general 
classifier based on training data from multiple missions. 
This folder is for development purposes only.

There are two parts to the code.

## Part I
Extracting the features from a RGB geotif file and a chm shapefile using the label geodatabase. This work is done in the features notebook, auxiliary functions are stored in data_extraction.py and preprocessing_aux.py. The extracted features are first stored in a pandas dataframe, then saved in a csv file.
File for Part I:
- features.ipynb
- data_extraction.py
- preprocessing_aux.py

## Part II
The features are now used to train a random forest classifier. Several training approaches are possible. The in_site approach trains one classifier per site (train_insite_RF.ipynb) which has a better performance in most cases. The generic approach uses features from all sites to train one generic classifier for all sites and which, ideally, is generalizable to other drone data collected in southern NWT (train_experiments.ipynb). Several experiments were run to train a generic classifier using different data subsets, features, or data processing steps with the aim to improve the final generic classifier performance. The auxiliary functions are stored in training.py.
Files for Part II:
- train_insite_RF.ipynb
- train_experiments.ipynb
- training.py
