# -*- coding: utf-8 -*-
"""
Created on Wed July 25 13:20:48 2024

@author: Matthew Fortier, Maria Belke Brea
This module is used for creating a general lichen classifier for highres UAV data (0.5 - 5 cm) 
collected in NWT, Canada.
The functions in this module train a Random Forest classifier (RFC)
based on a training dataset created with the modules data_extraction and preprocessing_aux.
The function 'in_site_RF' trains an in-site RFC. 
The functions 'site_cross_validation' and 'random_search' allow to set up experiments to
find the best performing general RFC for all sites. 

"""
from sklearn.model_selection import train_test_split
import imblearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
import random
from pylab import mean
from collections import Counter
import numpy as np

def split_xy(df, non_predictive_columns=[]):
    """
    Splits a dataset into features used for training (X) and labels (y). 
    Columns that are not used for classifier development are dropped 
    (e.g. 'x_pos', 'y_pos', 'site', 'class_certainty', 'veg_class').
    
    Parameters:
    -----------
    df: geopandas dataframe
    Dataframe containing labels as well as features used for classifier training.
    
    non_predictive_columns: list
    Columns that are not used for classifier training.
    Example: ['x_pos', 'y_pos', 'site', 'class_certainty', 'veg_class']
    
    Returns:
    --------
    X: geopandas dataframe
    Subset of df containing training features only.
    
    y: geopandas dataframe
    Subset of df containing labels only.
    
    """
    feature_columns = [c for c in list(df.columns) if c not in non_predictive_columns]
    X = df[feature_columns]
    y = df['veg_class']
    return X, y

# TODO: deleted split_xy argument, check if ok
#def split_by_sites(df, sites, split_xy=True, non_predictive_columns=[]):
def split_by_sites(df, sites, non_predictive_columns=[]):
    """
    More involved splitting than in split_xy(), used when prototyping experiments.
    Splits a dataset into features used for training (X) and labels (y) and 
    sets sites aside to be used as test sites. 
    Columns that are not used for classifier development are dropped 
    (e.g. 'x_pos', 'y_pos', 'site', 'class_certainty', 'veg_class').
    
    Parameters:
    -----------
    df: geopandas dataframe
    Dataframe containing labels as well as features used for classifier training.
    
    sites: list
    all sites given in the 'sites' argument are going to be test sites
    
    non_predictive_columns: list, default []
    Columns that are not used for classifier training.
    Example: ['x_pos', 'y_pos', 'site', 'class_certainty', 'veg_class']
    
    Returns:
    --------
    X_train: geopandas dataframe
    Subset of df containing training features only, excludes data from sites in 
    'site' argument.
    
    X_test: geopandas dataframe
    Subset of df containing data from test sites only (as specified in the 'site' argument)
    
    y_train: geopandas dataframe
    Subset of df containing labels only, excludes data from sites in 'site' argument.
    
    y_test: geopandas dataframe
    Subset of df containing labels from test sites only (as specified in the 'site' argument)
    
    """
    site_indices = df['site'].isin(sites)

    feature_columns = [c for c in list(df.columns) if c not in non_predictive_columns]
    X = df[feature_columns]
    y = df['veg_class']
    
    X_train = X[~site_indices].reset_index(drop=True)
    X_test = X[site_indices].reset_index(drop=True)
    y_train = y[~site_indices].reset_index(drop=True)
    y_test = y[site_indices].reset_index(drop=True)

    return X_train, X_test, y_train, y_test


def certainty_weight(df):
    """
    Weighs a training dataset based on label certainty. Certainty classes can be
    1 (very certain), 2 (mostly certain), 3 (less certain) and 4 (not certain). 
    The certainty column has to be labeled 'class_certainty'. 
    For weighing, class 1 labels are multiplied by 4, class 2 by 3, class 3 by 2, class 4 by 1.
    
    Parameters:
    -----------
    df: geopandas dataframe
    Dataframe containing a class certainty column (called 'class_certainty') 
    as well as labels and features used for classifier training.
    
    Returns:
    --------
    df_weighted: geopandas dataframe
    Variation of given dataframe ('df') where occurances of class 1, 2 and 3 labels
    have been multiplied to give them more weight.
    
    """
    cert1_indices = df['class_certainty'].isin([1])
    cert2_indices = df['class_certainty'].isin([2])
    cert3_indices = df['class_certainty'].isin([3])
    cert4_indices = df['class_certainty'].isin([4])

    df_cert1 = pd.concat([df[cert1_indices].reset_index(drop=True)]*4, ignore_index=True)
    df_cert2 = pd.concat([df[cert2_indices].reset_index(drop=True)]*3, ignore_index=True)
    df_cert3 = pd.concat([df[cert3_indices].reset_index(drop=True)]*2, ignore_index=True)
    df_cert4 = pd.concat([df[cert4_indices].reset_index(drop=True)]*1, ignore_index=True)

    df_weighted = pd.concat([df_cert1, df_cert2, df_cert3, df_cert4], ignore_index=True)
    
    return df_weighted

# Used for single site classifiers
def in_site_RF(df, site, test_size=0.33, extra_metrics=False, non_predictive_columns=[]):
    """
    Trains an random forest classifier using training data from one site only. 
    All classes but majority are resampled to do deal with label imbalance.
    #TODO
    Training parameters are hard coded right now (n_estimators=600, max_depth=6, n_jobs=12).
    
    Parameters:
    -----------
    df: geopandas dataframe
    Dataframe containing training features and labels ('veg_class' column) as well as a 'site' column.
    
    site: str
    Site name for RF training.
    
    test_size: float, default 0.33
    Split proportion for training and testing data.
    
    extra_metrics: bool, default False
    If set to true returns a dictionary containing measures of performance for the classifier 
    ('cf_matrix', 'feature_importance', 'train_accuracy', 'test_accuracy').
    
    non_predictive_columns: list, default empty list
    Columns that are not used for classifier training.
    Example: ['x_pos', 'y_pos', 'site', 'class_certainty', 'veg_class']
    
    Returns:
    --------
    clf: scikit RF model object
    Trained model algorithm
    
    preds: array
    Predicted classes from test dataset. 
    
    extra_metrics: dictionary
    dictionary containing measures of performance for the classifier 
    ('cf_matrix', 'feature_importance', 'train_accuracy', 'test_accuracy')
    
    """
    print(f'Processing {site}...')
    df_site = df[df['site'] == site]
    # df_reduced = df_site[df_site['class_certainty'] <= 2]
    X, y = split_xy(df_site, non_predictive_columns=non_predictive_columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # -------------------------------
    # oversample w synthetic minority oversampling to balance classes
    k_neighbors = 5
    c = Counter(y_train) #counts instances per class
    n_sample_min = min(c.values()) #least abundant class
    
    #set k_neighbors argument (resample function) to fit min abundance class
    if 1 < n_sample_min < 6 :
        k_neighbors = n_sample_min-1
    elif n_sample_min == 1:
        
        # min class == 1 raises k_neighbors error
        # duplicate line in y_train and X_train so n-sample of least common class is 2 (not 1)
        
        class_leastcommon = c.most_common()[:-2:-1][0][0] # get class number of least common class (eg 1 for lichen, 9 for moss, ...)
        row_index = pd.Index(y_train).get_loc(class_leastcommon) # get index for row with least common class
        y_train = pd.concat([y_train, y_train.iloc[[row_index]]]) # duplicate row
        X_train = pd.concat([X_train, X_train.iloc[[row_index]]]) # duplicate row
        
        k_neighbors=1 # setting k_neighbors to 1 will no longer throw an error because n-sample is 2 now
        
    under = imblearn.over_sampling.SMOTE(sampling_strategy='not majority', k_neighbors=k_neighbors) # resamples all classes but majority
    X_res, y_res = under.fit_resample(X_train, y_train)

    # -----------------------------------------------------------------------------------------------------------
    # Train RF classifier
    clf=RandomForestClassifier(n_estimators=600, max_depth=6, n_jobs=12)
    clf.fit(X_res.values, y_res.values)

    # -----------------------------------------------------------------------------------------------------------
    # Prediction
    preds = clf.predict(X_test.values)

    #print(f'  Train accuracy: {clf.score(X_res, y_res)}')
    #print(f'  Test accuracy: {clf.score(X_test, y_test)}')

    if extra_metrics:
        # -----------------------------------------------------------------------------------------------------------
        # Inspect performance parameters: training and testing scores, confusion matrix, and feature importanceax1 = plt.figure(figsize=(5,5)).add_subplot(111)
        extra_metrics = {}

        extra_metrics['cf_matrix'] = confusion_matrix(y_test, preds,  labels = clf.classes_)
        #disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, display_labels=clf.classes_)
        #disp.plot()
        #plt.title('Confusion Matrix')

        extra_metrics['feature_importance'] = pd.DataFrame(clf.feature_importances_, index=X_res.columns).sort_values(by=0, ascending=False)
        #print(feature_importance)
        
        extra_metrics['train_accuracy'] = clf.score(X_res, y_res)
        extra_metrics['test_accuracy'] = clf.score(X_test, y_test)
        
        return clf, extra_metrics, preds
    
    else:
        return clf, preds

# Run all CV splits and return results
def site_cross_validation(model, data, run_function, cut_size, test_sites, non_predictive_columns):
    '''
    This function uses cross validation to train a model of predefined type and hyperparameters. 
    The size of the cross-validation clusters is set with the 'cut_size' argument. Data from one site
    is either set aside for validation or used for training, not split up. 
    This function is called by the function 'random_search', where hyperparameters are defined. 
    Training and testing accuracies for each corss-validation loop are recorded and returned
    to 'random_search' to determine the best hyperparameter settings.
    
    Parameters:
    -----------
    model: scikit model object
    Build but untrained model object with defined model type and hyperparameters (defined in the random search function).
    Ready to receive training data.
    
    data: geopandas dataframe
    Dataframe containing training features and labels ('veg_class' column) as well as a 'site' column.
    
    run_function: function, user defined
    'run_function' is used to over / under sample to get around class imbalance
    
    cut_size: int
    Size for cross-validation clusters. For example cut_size=2 will set 2 sites 
    aside for cross validation.
    
    test_sites: list
    Sites in this list will not be used for model training, only for testing model
    performance.
    
    non_predictive_columns: list, default empty list 
    Columns that are not used for classifier training.
    Example: ['x_pos', 'y_pos', 'site', 'class_certainty', 'veg_class']
    
    Returns:
    --------
    train_accuracies: list
    lists train accuracies for each cross validation.
    
    test_accuracies: list
    lists test accuracies for each cross validation.
    '''
    train_accuracies = []
    val_accuracies = []
    
    data_sites = data['site'].unique()
    train_sites = [s for s in data_sites if s not in test_sites]
    random.shuffle(train_sites)
    cv_cuts = [train_sites[i:i + cut_size] for i in range(0, len(train_sites), cut_size)]
    #'cv_cuts' is a list of lists with site names used for cross validation
    #      ex: [ ['CS3A', 'ZF20-11A'], ['F3-20A', 'CG1-8A'], ... ]
    
    for cut in cv_cuts:
        train_data = data[~data['site'].isin(cut)]
        val_data = data[data['site'].isin(cut)]
        tr, va = run_function(model, train_data, val_data, non_predictive_columns) # this is where experiment fn is called
        train_accuracies.append(tr)
        val_accuracies.append(va)
        
    return train_accuracies, val_accuracies

# Random search procedure
def random_search(data, model_class, param_dist, run_function, n_iterations=50, cut_size=4, n_jobs=8, test_sites=[], non_predictive_columns=[]):
    '''
    This function receives a search space for model hyperparameters and calculates the 
    best-performing hyperparameter setting. The 'random_search' function was designed to work 
    with the RandomForestClassifier model class, but should also work with other
    scikit model classes. Model performance is determined for 'n_iterations', and 
    for each iteration a set of randomly selected hyperparameters is passed to 
    a 'cross_validation' function where model performance is evaluated. 
                
    Parameters:
    -----------
    data: geopandas dataframe
    The dataframe contains all the data the model is train on
    
    model_class: scikit model class
    The model class used for training. For example RandomForestClassifier
    
    param_dist: dictionary 
    Dictionary with a search space for each hyperparameter that should be optimized in this 'random_search' function.
    
    run_function: custom function
    custom function implementing experiment logic must have function signature:
            run_function(model, train_data, val_data) -> train_accuracy, val_accuracy
            
    n_iterations: int, default 50
    Number of iterations run to find best hyperparamter settings. 
    
    cut_size: int, default 4
    Argument that is passed on to the 'cross_validation' function.
    
    n_jobs: int, default 8
    
    test_sites: list, default empty list
    Argument is passed on to the 'cross_validation' function
    
    non_predictive_columns: list, default empty list
    Argument is passed on to the 'cross_validation' function
    
    Returns:
    --------
    None, only prints the best parameters and the associated best performance.
    '''
    
    best_params = None
    best_accuracy = 0.0
    
    for i in range(n_iterations):
        params = {key: random.choice(value) for key, value in param_dist.items()}
        model = model_class(n_jobs=n_jobs, **params)
        print(f'Trying {params}')
        train_accuracies, val_accuracies = site_cross_validation(model, data, run_function, cut_size, test_sites, non_predictive_columns)
        
        print(f'  Val accuracies: {val_accuracies}')
        print(f'  Train accuracy mean: {mean(train_accuracies)}, Val accuracy mean: {mean(val_accuracies)}')
        if mean(val_accuracies) > best_accuracy:
            best_accuracy = mean(val_accuracies)
            best_params = params
    
    print(f'\nBest params: {best_params}. Average test accuracy: {best_accuracy}')
    return best_params
    
def model_best_params(data, test_sites, non_predictive_columns, best_params, model_class=RandomForestClassifier, n_jobs=8):
    # train and validation data
    train_data = data[~data['site'].isin(test_sites)]
    X_train, y_train = split_xy(train_data, non_predictive_columns)

    val_data = data[data['site'].isin(test_sites)]
    X_val, y_val = split_xy(val_data, non_predictive_columns)

    # model set-up and model training
    n_jobs = 8
    model_architecture = model_class(n_jobs=n_jobs, **best_params)
    model = model_architecture.fit(X_train.values, y_train.values)

    # predicted values
    y_train_pred = model.predict(X_train.values)
    y_val_pred = model.predict(X_val.values)

    # dictionary
    ana_dict = {}
    ana_dict['test_sites'] = test_sites
    ana_dict['model'] = model
    ana_dict['train_values'] = {'X': X_train, 'y_true': y_train, 'y_pred': y_train_pred} 
    ana_dict['val_values'] = {'X': X_val, 'y_true': y_val, 'y_pred': y_val_pred}
    
    return(ana_dict)

def recall_precision(cfm):
    recall = []
    precision = []
    
    for line_count in range(0, len(cfm)):
        #line_rec = cfm[line_count]
        #correct_rec = line_rec[line_count]
        recall.append(cfm[line_count][line_count]/sum(cfm[line_count]))
        
        precision.append(cfm.T[line_count][line_count]/sum(cfm.T[line_count]))
        
    return recall, precision

def cfm_relativ(cfm):
    cfm_rel = []
    
    samples_per_class = [sum(i) for i in cfm]
    for i, line in enumerate(cfm):
        line_relativ = [round(val/samples_per_class[i], 2)*100 for val in line]
        cfm_rel.append(line_relativ)
    
    return np.array(cfm_rel)