# -*- coding: utf-8 -*-
"""
Created on Sun May  7 18:10:58 2023

@author: MABEB16
"""

import pandas as pd


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
