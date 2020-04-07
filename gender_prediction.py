import pandas as pd
import numpy as np

# Create a function to edit data set. 
def add_Column(data):
    data["Hyp"] = 0
    data["Results"] = 0


def make_Hyp(data):
    data.loc[data["Sex"] == "F", "Hyp"] = 1


def Results(data):
    data.loc[data["Survived"] == data["Hyp"],"Results"] = 1
    pred = data["Results"].value_counts(normalize=True)
    print(pred)

files = ["test_cleaned.csv"]

for file in files:
    data = pd.read_csv(file)
    add_Column(data)
    make_Hyp(data)
    Results(data)