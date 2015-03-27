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
from sklearn.ensemble import ExtraTreesClassifier

# Feature selection
titanic_data = pd.concat( [ titanic_data, pd.get_dummies( titanic_data[ [ 'Sex' ] ] )  ], axis = 1 )
titanic_data = pd.concat( [ titanic_data, pd.get_dummies( titanic_data[ [ 'Embarked' ] ] )  ], axis = 1 )
titanic_data['Embarked'] = titanic_data['Embarked'].replace('S', 0).replace('C', 1).replace('Q', 2)
titanic_data['FamilySize'] = titanic_data['SibSp'] + titanic_data['Parch']
titanic_data['Alone'] = 1
titanic_data.ix[ titanic_data['FamilySize'] > 0, 'Alone' ] = 0
titanic_data['FareBins'] = 0
titanic_data.ix[ titanic_data['Fare'] <= 10, 'FareBins' ] = 1
titanic_data.ix[ (titanic_data['Fare'] <= 40) & (titanic_data['Fare'] > 10), 'FareBins' ] = 2
titanic_data.ix[ titanic_data['Fare'] >= 40, 'FareBins' ] = 3
titanic_data['Title'] = titanic_data['Name'].str.split().str.get(1)
titanic_data = pd.concat( [ titanic_data, pd.get_dummies( titanic_data[ [ 'Title' ] ] )  ], axis = 1 )
titanic_data = titanic_data.drop('Cabin', 1)

# ['Mr.' 'Mrs.' 'Miss.' 'Master.' 'Planke,' 'Don.' 'Rev.' 'Billiard,' 'der'
#  'Walle,' 'Dr.' 'Pelsmaeker,' 'Mulder,' 'y' 'Steen,' 'Carlo,' 'Mme.'
#  'Impe,' 'Ms.' 'Major.' 'Gordon,' 'Messemaeker,' 'Mlle.' 'Col.' 'Capt.'
#  'Velde,' 'the' 'Shawah,' 'Jonkheer.' 'Melkebeke,' 'Cruyssen,']
# titanic.py:52: DataConversionWarning: A column-vector y was pass

# Feature selection for test set
titanic_data_test = pd.concat( [ titanic_data_test, pd.get_dummies( titanic_data_test[ [ 'Sex' ] ] )  ], axis = 1 )
titanic_data_test = pd.concat( [ titanic_data_test, pd.get_dummies( titanic_data_test[ [ 'Embarked' ] ] )  ], axis = 1 )
titanic_data_test['Embarked'] = titanic_data_test['Embarked'].replace('S', 0).replace('C', 1).replace('Q', 2)
titanic_data_test['FamilySize'] = titanic_data_test['SibSp'] + titanic_data_test['Parch']
titanic_data_test['FareBins'] = 0
titanic_data_test.ix[ titanic_data_test['Fare'] <= 10, 'FareBins' ] = 1
titanic_data_test.ix[ (titanic_data_test['Fare'] <= 40) & (titanic_data_test['Fare'] > 10), 'FareBins' ] = 2
titanic_data_test.ix[ titanic_data_test['Fare'] >= 40, 'FareBins' ] = 3
titanic_data_test['Title'] = titanic_data_test['Name'].str.split().str.get(1)
titanic_data_test = pd.concat( [ titanic_data_test, pd.get_dummies( titanic_data_test[ [ 'Title' ] ] )  ], axis = 1 )
titanic_data_test = titanic_data_test.drop('Cabin', 1)

# Need to fill in empty age data with a model of the other variables
age_model_features = ['FareBins', 'Pclass', 'Title_Mr.', 'Title_Mrs.', 'Title_Miss.', 
				'Title_Master.', 'Title_Dr.',  'Title_Ms.',  
				'Title_Col.', 'Sex_male']
X_age_model = titanic_data.dropna()[age_model_features]
y_age_model = titanic_data.dropna()[['Age']]

from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(X_age_model, y_age_model)
y_age_predicted = rfr.predict(titanic_data[pd.isnull(titanic_data['Age']) == True][age_model_features])
y_age_predicted_df = pd.DataFrame(y_age_predicted)
age_na_index = titanic_data.ix[pd.isnull(titanic_data['Age']) == True, 'Age'].index
y_age_predicted_df.index = age_na_index
for index, val in y_age_predicted_df.iterrows():
	titanic_data.ix[index, 'Age'] = y_age_predicted_df.ix[index, 0]


y_age_predicted_test = rfr.predict(titanic_data_test[pd.isnull(titanic_data_test['Age']) == True][age_model_features])
y_age_predicted_test_df = pd.DataFrame(y_age_predicted_test)
age_na_index = titanic_data_test.ix[pd.isnull(titanic_data_test['Age']) == True, 'Age'].index
y_age_predicted_test_df.index = age_na_index
for index, val in y_age_predicted_test_df.iterrows():
	titanic_data_test.ix[index, 'Age'] = y_age_predicted_test_df.ix[index, 0]

# Model features
features = [ 'Sex_male', 'FareBins', 'Pclass', 'Parch', 'SibSp', 'Age','Title_Mr.', 'Title_Mrs.', 'Title_Miss.', 
				'Title_Master.', 'Title_Dr.',  'Title_Ms.',  
				'Title_Col.' ]
titanic_data_model = titanic_data[ features  + ['Survived'] ].fillna(titanic_data[ features ].median())
#titanic_data_model = titanic_data[ features  + ['Survived'] ]
titanic_data_feats_X = titanic_data_model.drop('Survived', 1)
titanic_data_Y = titanic_data_model['Survived']

# # Fill NaNs for test data
titanic_data_test_feats_X = titanic_data_test[ features ].fillna(titanic_data[ features ].median())

# Train test split
# skf = cval.StratifiedKFold(titanic_data_Y, n_folds=5)
# x_train = []
# x_test = []
# y_train = []
# y_test = []
# for train_index, test_index in skf:
# 	x_train.append(titanic_data_feats_X.iloc[train_index])
# 	x_test.append(titanic_data_feats_X.iloc[test_index])
# 	y_train.append(titanic_data_Y.iloc[train_index])
# 	y_test.append(titanic_data_Y.iloc[test_index])
x_train, x_test, y_train, y_test = cval.train_test_split(titanic_data_feats_X, titanic_data_Y, train_size=0.6)


# Models
classifiers = {
	'random_forest' : RandomForestClassifier(max_depth=4, n_estimators=2000),
	'gradient_boost' : GradientBoostingClassifier( learning_rate = 0.3, n_estimators = 2000, max_depth=3 ),
	'extra_trees' : ExtraTreesClassifier(n_estimators=2000, max_depth=4, min_samples_split=10, min_samples_leaf=10),
	'naive_bayes' : GaussianNB(),
	'logit' : LogisticRegression(C = 0.1)
}

classifier_f1 = {
	'random_forest' : [],
	'gradient_boost' : [],
	'extra_trees' : [],
	'naive_bayes' : [],
	'logit' : [],
}

# Fit with k folds
# for kfold in range(len(x_train)):
for classifier in classifiers:
	classifiers[classifier].fit(x_train, y_train)
	classifier_f1[classifier].append(metrics.accuracy_score(y_test, classifiers[classifier].predict(x_test)))

# Output f1 scores
for classifier in classifiers:
	print classifier
	print np.mean(classifier_f1[classifier])

# Fit full training set
for classifier in classifiers:
	classifiers[classifier].fit(titanic_data_feats_X, titanic_data_Y)

# # Test on new data
#print classifiers['logit'].feature_importances_
predicted_values = pd.DataFrame(classifiers['random_forest'].predict(titanic_data_test_feats_X))
predicted_values.columns = ['Survived']
return_data = pd.concat( [ titanic_data_test[ ['PassengerId'] ], predicted_values ], axis=1)
return_data.to_csv('results.csv', index=False)


