import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def log_normal_nll(y, mu, sigma):
    logy = torch.log(y + 1e-8)  # avoid log(0)
    nll = torch.log(y + 1e-8) + 0.5 * torch.log(2 * torch.pi * sigma**2) \
          + 0.5 * ((logy - mu) ** 2) / (sigma**2)
    return nll.mean()

# --- Load data ---
X_train = pd.read_csv('X_trn.csv')
y_train = pd.read_csv('Y_trn.csv')

# --- 1. Opleiding (ordinaal) ---
educcat_categories = [["Less Than High School", 
                       "High School", 
                       "Junior College", 
                       "Bachelor", 
                       "Graduate"]]
educcat_encoder = OrdinalEncoder(categories=educcat_categories)
X_train["educcat"] = educcat_encoder.fit_transform(X_train[["educcat"]]).astype(int)

# --- 2. One-hot encode voor kleine categoricals ---
onehot_cols = ["gender", "maritalcat", "wrkstat"]
X_train = pd.get_dummies(X_train, columns=onehot_cols, drop_first=True)

# --- 3. Encode occrecode (voor embeddings later in NN) ---
# Maak integer labels die je in een embedding layer kunt gebruiken
occrecode_mapping = {cat: idx for idx, cat in enumerate(X_train["occrecode"].unique())}
X_train["occrecode"] = X_train["occrecode"].map(occrecode_mapping)

# --- 4. Standaardiseer numerieke variabelen ---
numeric_cols = ["year", "age", "prestg10", "childs"]
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])

# --- Check resultaat ---
print(X_train.head())
print("Educcat:", X_train["educcat"].unique())
print("Occrecode range:", X_train["occrecode"].min(), "-", X_train["occrecode"].max())

# --- 5. Split in training en validatie set ---
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)
