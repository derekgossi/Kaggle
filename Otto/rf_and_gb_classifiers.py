
# Import data
import pandas as pd
import pickle

with open('pickle/processed_df.p', mode='rb') as picklefile:
      df = pickle.load(picklefile)

# Feature set
features = df[df["test"] == 0].iloc[:,0:93]
labels = df[df["test"] == 0].iloc[:,94]

## Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Random forest
rf_model = RandomForestClassifier(n_estimators=10, criterion='gini', 
                                  max_depth=None, min_samples_split=2, 
                                  min_samples_leaf=1, max_features='auto', 
                                  max_leaf_nodes=None, bootstrap=True, 
                                  oob_score=False, n_jobs=1, 
                                  random_state=None, verbose=0, 
                                  min_density=None, compute_importances=None)

#rf_model_fitted = rf_model.fit(X_train, y_train)

# Gradient boosted trees
gb_model = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, 
                                      n_estimators=300, subsample=0.75, 
                                      min_samples_split=20, min_samples_leaf=5, 
                                      max_depth=6, init=None, random_state=None, 
                                      max_features='auto', verbose=1, 
                                      max_leaf_nodes=None, warm_start=False)

#gb_model_fitted = gb_model.fit(X_train, y_train)

# Model validation
from sklearn import cross_validation
import numpy as np

# # Random forest classifier
# rf_score = cross_validation.cross_val_score(rf_model, features, labels, scoring='log_loss', cv=3)
# print "Random forest CV score", np.mean(rf_score)

# # Gradient boosted classifier
# gb_score = cross_validation.cross_val_score(gb_model, features_samp, labels_samp, scoring='log_loss', cv=5)
# print "Gradient boosting CV score", np.mean(gb_score)

# Gradient boosted classifier - submission
gb_fitted = gb_model.fit(features, labels)

# Predict test values
features_test = df[df["test"] == 1].iloc[:,0:93]
gb_pred = gb_fitted.predict_proba(features_test)

## Make submission
import csv

file_h = 'submission/submission.csv'

with open(file_h, 'wb') as csv_file:
    csv_w = csv.writer(csv_file)
    csv_w.writerow(['id', 'Class_1', 'Class_2', 'Class_3', 
                   'Class_4', 'Class_5', 'Class_6',
                   'Class_7', 'Class_8', 'Class_9'])
    
    for i, pred_vals in enumerate(gb_pred):
        csv_w.writerow([i+1] + list(pred_vals))

