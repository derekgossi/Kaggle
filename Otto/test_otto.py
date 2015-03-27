import pandas as pd

# Read training data
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

## Data preprocessing

# Integer feature names
train_df.columns = ["id"] + range(1,94) + ["target"]
test_df.columns = ["id"] + range(1,94)

# Add train and test column
train_df["test"] = 0
test_df["test"] = 1

# Change target value to integer class
for old_class in train_df["target"].unique():
    new_class = list(old_class)[-1:]
    new_class = int(new_class[0])
    train_df["target"] = train_df["target"].replace(old_class, new_class)

# Merge two dataframes
df = pd.concat([train_df, test_df])

# Feature set
features = df[df["test"] == 0].iloc[:,0:93]
labels = df[df["test"] == 0].iloc[:,94]

# Train test split for training set
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)

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
gb_model = GradientBoostingClassifier(loss='deviance', learning_rate=0.2, 
                                      n_estimators=500, subsample=1.0, 
                                      min_samples_split=2, min_samples_leaf=1, 
                                      max_depth=5, init=None, random_state=None, 
                                      max_features=None, verbose=0, 
                                      max_leaf_nodes=None, warm_start=False)

#gb_model_fitted = gb_model.fit(X_train, y_train)

# Model validation
from sklearn import cross_validation
import numpy as np

# Random forest classifier
rf_score = cross_validation.cross_val_score(rf_model, features, labels, scoring='log_loss', cv=3)
print "Random forest CV score", np.mean(rf_score)

# Gradient boosted classifier
gb_score = cross_validation.cross_val_score(gb_model, features, labels, scoring='log_loss', cv=3)
print "Gradient boosting CV score", np.mean(gb_score)

