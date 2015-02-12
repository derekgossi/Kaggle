import pandas as pd
import numpy as np
import csv

# Read data from csv
titanic_data = pd.read_csv('data/train.csv')
titanic_data_test = pd.read_csv('data/test.csv')

## Baseline model

from sklearn import cross_validation as cval
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# Feature selection
titanic_data = pd.concat( [ titanic_data, pd.get_dummies( titanic_data[ [ 'Sex' ] ] )  ], axis = 1 )
titanic_data['Embarked'] = titanic_data['Embarked'].replace('S', 0).replace('C', 1).replace('Q', 2)
titanic_data['FamilySize'] = titanic_data['SibSp'] + titanic_data['Parch']
titanic_data['Alone'] = 1
titanic_data.ix[ titanic_data['FamilySize'] > 0, 'Alone' ] = 0 
titanic_data['Age*Class'] = titanic_data.Age * titanic_data.Pclass
titanic_data['FareBins'] = 0
titanic_data.ix[ titanic_data['Fare'] <= 10, 'FareBins' ] = 1
titanic_data.ix[ (titanic_data['Fare'] <= 40) & (titanic_data['Fare'] > 10), 'FareBins' ] = 2
titanic_data.ix[ titanic_data['Fare'] >= 40, 'FareBins' ] = 3


# Feature selection for test set
titanic_data_test = pd.concat( [ titanic_data_test, pd.get_dummies( titanic_data_test[ [ 'Sex' ] ] )  ], axis = 1 )
titanic_data_test['Embarked'] = titanic_data_test['Embarked'].replace('S', 0).replace('C', 1).replace('Q', 2)
titanic_data_test['FamilySize'] = titanic_data_test['SibSp'] + titanic_data_test['Parch']
titanic_data_test['Age*Class'] = titanic_data_test.Age * titanic_data_test.Pclass

# Model features
features = [ 'Sex_male', 'FamilySize', 'Age*Class' ]
titanic_data_model = titanic_data[ features  + ['Survived'] ].fillna(titanic_data[ features ].median())
titanic_data_feats_X = titanic_data_model.drop('Survived', 1)
titanic_data_Y = titanic_data_model['Survived']

# Fill NaNs for test data
titanic_data_test_feats_X = titanic_data_test[ features ].fillna(titanic_data[ features ].median())

# Train test split
skf = cval.StratifiedKFold(titanic_data_Y, n_folds=15, shuffle=True)
x_train = []
x_test = []
y_train = []
y_test = []
for train_index, test_index in skf:
	x_train.append(titanic_data_feats_X.iloc[train_index])
	x_test.append(titanic_data_feats_X.iloc[test_index])
	y_train.append(titanic_data_Y.iloc[train_index])
	y_test.append(titanic_data_Y.iloc[test_index])

# Models
classifiers = {
	'random_forest' : RandomForestClassifier(),
	'gradient_boost' : GradientBoostingClassifier( learning_rate = 0.4, n_estimators = 200 ),
	'gradient_boost2' : GradientBoostingClassifier( learning_rate = 0.2, n_estimators = 200 ),
	'gradient_boost3' : GradientBoostingClassifier( learning_rate = 0.1, n_estimators = 200 ),
	'naive_bayes' : GaussianNB(),
	'logit' : LogisticRegression(C = 10.0)
}

classifier_f1 = {
	'random_forest' : [],
	'gradient_boost' : [],
	'gradient_boost2' : [],
	'gradient_boost3' : [],
	'naive_bayes' : [],
	'logit' : [],
	'logit2' : [],
	'logit3' : [],
	'logit4' : []
}

# Fit with k folds
for kfold in range(len(x_train)):
	for classifier in classifiers:
		classifiers[classifier].fit(x_train[kfold], y_train[kfold])
		classifier_f1[classifier].append(metrics.accuracy_score(y_test[kfold], classifiers[classifier].predict(x_test[kfold])))

# Output f1 scores
for classifier in classifiers:
	print classifier
	print np.mean(classifier_f1[classifier])

# # Fit full training set
# for classifier in classifiers:
# 	classifiers[classifier].fit(titanic_data_feats_X, titanic_data_Y)

# # Test on new data
# #print classifiers['logit'].feature_importances_
# predicted_values = pd.DataFrame(classifiers['gradient_boost3'].predict(titanic_data_test_feats_X))
# predicted_values.columns = ['Survived']
# return_data = pd.concat( [ titanic_data_test[ ['PassengerId'] ], predicted_values ], axis=1)
# return_data.to_csv('results.csv', index=False)


