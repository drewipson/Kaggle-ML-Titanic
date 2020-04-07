import pandas as pd

test = pd.read_csv('test_cleaned.csv')

test[['PassengerId', 'Sex']].to_csv('gender_scored.csv', index=False)