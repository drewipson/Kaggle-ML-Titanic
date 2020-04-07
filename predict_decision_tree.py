# Drew Ipson | Decision Tree ML Model | March 12, 2020
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from sklearn import tree
# Read in csv files and store and data frames. Feature_values outlines which labels will be used to train model. 
train = pd.read_csv('train_cleaned.csv')
test_data = pd.read_csv('test_cleaned.csv')
target = train['Survived'].values
feature_values = ['Pclass', 'Embarked', 'Fare', 'Age', 'Sex', 'Deck_Index', 'Title_Index', 'Age_Class', 'Parch', 'SibSp']
features = train[feature_values].values

# Set random_state equal to 1 to have a random start in the decision tree classifier 
# to avoid decision making based off of derivatives
decision_tree = tree.DecisionTreeClassifier(max_depth=1, min_samples_split=2, random_state = 1)
decision_tree_ = decision_tree.fit(features, target)
print('Decision Tree Accuracy: ' + str(decision_tree_.score(features, target)))

# Make data prediciton based on training model:
test = test_data[feature_values].values
prediction = decision_tree.predict(test)
test_data['Survived'] = prediction
test_data[['PassengerId', 'Survived']].to_csv('prediction_values.csv',index=False)
