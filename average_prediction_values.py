# Drew Ipson | 3 ML Model | March 12, 2020
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from sklearn import tree, linear_model, preprocessing
# Read in csv files and store and data frames. Feature_values outlines which labels will be used to train model. 
train = pd.read_csv('train_cleaned.csv')
test_data = pd.read_csv('test_cleaned.csv')
target = train['Survived'].values
feature_values = ['Pclass', 'Embarked', 'Fare', 'Age', 'Sex', 'Deck_Index', 'Title_Index', 'Age_Class', 'Parch', 'SibSp']
features = train[feature_values].values
# Set random_state equal to 1 to have a random start in the decision tree classifier 
# to avoid decision making based off of derivatives
decision_tree = tree.DecisionTreeClassifier(max_depth=7, min_samples_split=2, random_state = 1)
decision_tree_ = decision_tree.fit(features, target)
print('Decision Tree Accuracy: ' + str(decision_tree_.score(features, target)))
# Make data prediciton based on training model:
test = test_data[feature_values].values
dt_prediction = decision_tree.predict(test)
# Lays out classifier using a Linear Model.
linear_ = linear_model.LogisticRegression()
linear_fit = linear_.fit(features, target)
print('Linear Classifier Score: ' + str(linear_.score(features, target)))
# Linear Predictions
test_data_features = test_data[feature_values].values
linear_pred = linear_fit.predict(test_data_features)
# Lays out classifier using polynomial transformations
poly = preprocessing.PolynomialFeatures(degree=2)
poly_features = poly.fit_transform(features)
poly_ = linear_.fit(poly_features, target)
print('Poly Classifier Score: ' + str(linear_.score(poly_features, target)))
# Poly predictions
test_poly_features = poly.fit_transform(test_data_features)
poly_pred = poly_.predict(test_poly_features)
# total matrix for data
total_matrix = (dt_prediction + linear_pred + poly_pred)/3
# setting matrix averages to 1 or 0
for i in range(len(total_matrix)):
    if (total_matrix[i] > .6):
        total_matrix[i] = 1
    else:
        total_matrix[i] = 0
# Output to CSV
test_data['Survived'] = total_matrix.astype('int64')
test_data[['PassengerId', 'Survived']].to_csv('average_predictions.csv',index=False)