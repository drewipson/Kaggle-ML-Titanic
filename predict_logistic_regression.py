# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from sklearn import linear_model, preprocessing
# Reads in CSV files to data frames and lays out which labels to use to train model.
train = pd.read_csv("train_cleaned.csv")
train = train.reset_index()
target = train["Survived"].values
feature_values = ["Pclass", "Embarked", "Fare", "Age", "Sex", "SibSp", "Parch", "Deck_Index", "Title_Index", "Age_Class"]
features = train[feature_values].values
# Lays out classifier using a Linear Model.
classifier = linear_model.LogisticRegression()
classifier_ = classifier.fit(features, target)
print('Linear Classifier Score: ' + str(classifier_.score(features, target)))
# Lays out classifier using polynomial model
poly = preprocessing.PolynomialFeatures(degree=10)
poly_features = poly.fit_transform(features)
classifier_ = classifier.fit(poly_features, target)
print('Poly Classifier Score: ' + str(classifier_.score(poly_features, target)))
# Output Predictions from Poly Classifier
test = pd.read_cv('test_cleaned.csv')
test_features = poly.fit_transform(features)