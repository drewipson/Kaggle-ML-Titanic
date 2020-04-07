# Drew Ipson | Feb 12, 2020 | Kaggle - Titanic Competition | UVU Data Science Club
# import pandas module for data frame manipulation and re for reg expression manipulation.
import pandas as pd
import re
# dictionaries for city reference
cities = {"C": "Cherbourg", "S": "Southampton", "Q": "Queenstown", "U": "Unknown"}
ports = {"Southampton" : 0, "Cherbourg" : 1, "Queenstown": 2, "Unknown": 3}
sex = {"M": 0, "F": 1}
decks = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "U": 7}
title = {'Miss': 0, 'Mr': 1, 'Mrs': 2, 'Ms': 3, 'Rare': 4}
# function to clean data and use .loc (module for read and writing)
# to clean up data errors in data set.
def clean_data(data, fileName):
    # Reassigns sex male and female to M and F.
    data.loc[data["Sex"] == "male", "Sex"] = "M"
    data.loc[data["Sex"] == "female", "Sex"] = "F"
    data["Sex"] = data["Sex"].map(sex)
    # fills NaN values with average to even distribution of data set.
    data["Fare"] = data["Fare"].fillna(data["Fare"].dropna().median())
    data["Age"] = data["Age"].fillna(data["Age"].dropna().median())
    # change city abbreviations from single letter to full name
    data["Embarked"] = data["Embarked"].fillna('U')
    data["Embarked"] = data["Embarked"].map(cities)
    data["Embarked"] = data["Embarked"].map(ports)
    # Organize data by deck level from cabin column. First: Create deck column and organize by deck level. Drop cabin column
    data["Cabin"] = data["Cabin"].fillna("U0")
    data["Deck"] = data["Cabin"].map(lambda x:re.compile("([a-zA-Z]+)").search(x).group())
    data["Deck_Index"] = data["Deck"].map(decks)
    data["Deck_Index"] = data["Deck_Index"].fillna('7')
    data = data.drop(["Cabin"],axis=1)
    # extract titles from names and assign values to female rows if they are married
    data["Title"] = data["Name"].str.extract(r', ([a-zA-z]+)\.', expand=False)
    data["Title"] = data["Title"].fillna('Unknown')
    data["Title"] = data["Title"].replace(['Capt', 'Col', 'Don', 'Dr','Lady', 'Major','Master', 'Rev', 'Sir','Unknown','Dona'], 'Rare')
    data["Title"] = data["Title"].replace(['Jonkheer'], 'Mr')
    data["Title"] = data["Title"].replace(['Mlle'], 'Miss')
    data["Title"] = data["Title"].replace(['Mme'], 'Mrs')

    data["Title_Index"] = data["Title"].map(title)
    # Create new features: AgeClass, which is the class multiplied by Age
    data["Age_Class"] = data["Age"] * data["Pclass"]
    # Modfiy data types of values. 
    data["Age"] = data["Age"].astype('int64')
    data["Deck_Index"] = data["Deck_Index"].astype('int64')
    data["Age_Class"] = data["Age_Class"].astype('int64')
    # save data frame to csv file.s
    data.to_csv(fileName[:-4] + "_cleaned.csv", index=False)  

# call function to run for reach data set.
fileName = ["train.csv", "test.csv"]

for file in fileName:
    data = pd.read_csv(file)
    clean_data(data, file)
