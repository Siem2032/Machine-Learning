import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder

X_train = pd.read_csv('X_trn.csv')
y_train = pd.read_csv('Y_trn.csv')

categories = [["Less Than High School", "High School", "Junior College", "Bachelor", "Graduate"]]
encoder = OrdinalEncoder(categories=categories)

X_train["education"] = encoder.fit_transform(X_train[["educcat"]]).astype(int)
encoder = LabelEncoder()

# Fit and transform
X_train = encoder.fit_transform(X_train[["gender", "city"]])
print(X_train['educcat'].head())
print(X_train['education'].head())


