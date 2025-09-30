import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder

X_train = pd.read_csv('X_trn.csv')
y_train = pd.read_csv('Y_trn.csv')

categories = [["Less Than High School", "High School", "Junior College", "Bachelor", "Graduate"]]
categorical_cols = ["occrecode", "gender", "maritalcat", "wrkstat"]
encoder = OrdinalEncoder(categories=categories)

X_train["educcat"] = encoder.fit_transform(X_train[["educcat"]]).astype(int)
print(X_train.head())
encoder = LabelEncoder()
for col in categorical_cols:
    X_train[col] = encoder.fit_transform(X_train[col]).astype(int)


# Fit and transform
print(X_train.head())
print(X_train['educcat'].head())



