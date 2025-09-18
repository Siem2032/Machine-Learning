import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

X_train = pd.read_csv('X_trn.csv')
y_train = pd.read_csv('Y_trn.csv')

encoder = LabelEncoder()
X_train["education"] = encoder.fit_transform(X_train["educcat"])
print(X_train.head())


