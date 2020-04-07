import pandas as pd
import numpy as np
import xgboost_decision_tree as xg_dt

train = pd.read_csv('train_cleaned.csv')
test_data = pd.read_csv('test_cleaned.csv')
target = train['Survived'].values
feature_values = ['Pclass', 'Embarked', 'Fare', 'Age', 'Sex', 'Deck_Index', 'Title_Index', 'Age_Class', 'Parch', 'SibSp']
train_features = train[feature_values].values