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
titanic_data['Age*Class'] = titanic_data.Age * titanic_data.Pclass

# Feature selection for test set
titanic_data_test = pd.concat( [ titanic_data_test, pd.get_dummies( titanic_data_test[ [ 'Sex' ] ] )  ], axis = 1 )
titanic_data_test['Embarked'] = titanic_data_test['Embarked'].replace('S', 0).replace('C', 1).replace('Q', 2)
titanic_data_test['FamilySize'] = titanic_data_test['SibSp'] + titanic_data_test['Parch']
titanic_data_test['Age*Class'] = titanic_data_test.Age * titanic_data_test.Pclass

# Model features
features = [ 'Pclass', 'Age', 'Sex_male' , 'Fare', 'FamilySize', 'Embarked' ]
titanic_data_model = titanic_data[ features  + ['Survived'] ].dropna()
titanic_data_feats_X = titanic_data_model.drop('Survived', 1)
titanic_data_Y = titanic_data_model['Survived']

# Fill NaNs for test data
titanic_data_test_feats_X = titanic_data_test[ features ].fillna(titanic_data[ features ].median())

# Train test split
x_train, x_test, y_train, y_test = cval.train_test_split(titanic_data_feats_X, titanic_data_Y, test_size = 0.4)

# Models
classifiers = {
	#'random_forest' : RandomForestClassifier(),
	'gradient_boost' : GradientBoostingClassifier( learning_rate = 0.2, n_estimators = 500 )
	# 'naive_bayes' : GaussianNB(),
	# 'logit' : LogisticRegression()

}

for classifier in classifiers:
	#classifiers[classifier].fit(titanic_data_feats_X, titanic_data_Y)
	classifiers[classifier].fit(x_train, y_train)
	print classifier 
	print metrics.classification_report(y_test, classifiers[classifier].predict(x_test))

# Test on new data
print classifiers['gradient_boost'].feature_importances_
predicted_values = pd.DataFrame(classifiers['gradient_boost'].predict(titanic_data_test_feats_X))
predicted_values.columns = ['Survived']
return_data = pd.concat( [ titanic_data_test[ ['PassengerId'] ], predicted_values ], axis=1)
return_data.to_csv('results.csv', index=False)


