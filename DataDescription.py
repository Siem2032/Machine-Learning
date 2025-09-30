import pandas as pd
import numpy as np

X = pd.read_csv("X_trn.csv")
y = pd.read_csv("y_trn.csv")  

# Lijst van categorische kolommen die je wil omzetten
categorical_cols = ["educcat", "occrecode", "gender", "maritalcat", "wrkstat"]

# Loop over alle categorische kolommen
for col in categorical_cols:
    print(f"\n=== {col.upper()} ===")
    print("Unieke waarden:", X[col].unique())
    
    # Zet om naar categorie-codes (1, 2, 3, ...)
    X[col] = X[col].astype("category").cat.codes + 1

print(X.head())
print(X.dtypes)

print('Real inc:',min(y["realrinc"]))


