import pandas as pd
from sklearn import tree
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

# Reading in files to data frame
df_train = pd.read_csv('train_cleaned.csv')
df_test = pd.read_csv('test_cleaned.csv')
# Set target values to variable
Y = df_train['Survived'].values
# Declare what features we want to feed into our DT model
features_values = ['Pclass', 'Embarked', 'Fare', 'Age', 'Sex', 'Deck_Index', 'Title_Index', 'Age_Class', 'Parch', 'SibSp']
X = df_train[features_values].values
# Machine Learning: Decision Tree - Training Model
clf = tree.DecisionTreeClassifier(max_depth=1,min_samples_split=2,random_state=1)
decision_tree = clf.fit(X,Y)
print('Decision Tree Training Accuracy: ' + str(decision_tree.score(X, Y)))
# Predict on our test data;
test_features = df_test[features_values].values
predictions = clf.predict(test_features)
df_test['Survived'] = predictions

df_test[['PassengerId', 'Survived']].to_csv('club_prediction_results.csv',index=False)