import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class IncomeDataset(Dataset):
    def __init__(self, df, y):
        self.numeric = torch.tensor(df[["year", "age", "prestg10", "childs"]].values, dtype=torch.float32)
        self.onehot = torch.tensor(df.drop(columns=["year","age","prestg10","childs","educcat","occrecode"]).values, dtype=torch.float32)
        self.educcat = torch.tensor(df["educcat"].values, dtype=torch.float32).unsqueeze(1)
        self.occrecode = torch.tensor(df["occrecode"].values, dtype=torch.long)
        self.y = torch.tensor(y.values.squeeze(), dtype=torch.float32)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.numeric[idx], self.onehot[idx], self.educcat[idx], self.occrecode[idx], self.y[idx]


class IncomeNN(nn.Module):
    def __init__(self, num_numeric_features, num_onehot_features, occrecode_cardinality, embedding_dim=16, hidden_dims=[128, 64]):
        super().__init__()
        
        # Embedding voor occrecode
        self.occrecode_emb = nn.Embedding(occrecode_cardinality, embedding_dim)
        
        # Totale inputdim = numerics + one-hot + ordinal (educcat) + embedding
        input_dim = num_numeric_features + num_onehot_features + 1 + embedding_dim  # +1 voor educcat
        
        # Hidden layers
        layers = []
        dim_in = input_dim
        for dim_out in hidden_dims:
            layers.append(nn.Linear(dim_in, dim_out))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            dim_in = dim_out
        self.hidden = nn.Sequential(*layers)
        
        # Output: mu en sigma
        self.mu_head = nn.Linear(dim_in, 1)
        self.sigma_head = nn.Linear(dim_in, 1)
    
    def forward(self, numeric, onehot, educcat, occrecode):
        occ_emb = self.occrecode_emb(occrecode)
        x = torch.cat([numeric, onehot, educcat, occ_emb], dim=1)
        h = self.hidden(x)
        mu = self.mu_head(h).squeeze(-1)
        sigma = F.softplus(self.sigma_head(h)).squeeze(-1) + 1e-6  # sigma > 0
        return mu, sigma

def log_normal_nll(y, mu, sigma):
    logy = torch.log(y + 1e-8)  # avoid log(0)
    nll = torch.log(y + 1e-8) + 0.5 * torch.log(2 * torch.pi * sigma**2) \
          + 0.5 * ((logy - mu) ** 2) / (sigma**2)
    return nll.mean()

def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3, patience=5, device="cpu"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float("inf")
    patience_counter = 0
    
    model.to(device)
    
    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss = 0
        for numeric, onehot, educcat, occrecode, y in train_loader:
            numeric, onehot, educcat, occrecode, y = numeric.to(device), onehot.to(device), educcat.to(device), occrecode.to(device), y.to(device)
            
            optimizer.zero_grad()
            mu, sigma = model(numeric, onehot, educcat, occrecode)
            loss = log_normal_nll(y, mu, sigma)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * y.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # --- Validation ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for numeric, onehot, educcat, occrecode, y in val_loader:
                numeric, onehot, educcat, occrecode, y = numeric.to(device), onehot.to(device), educcat.to(device), occrecode.to(device), y.to(device)
                mu, sigma = model(numeric, onehot, educcat, occrecode)
                loss = log_normal_nll(y, mu, sigma)
                val_loss += loss.item() * y.size(0)
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # --- Early stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    
    model.load_state_dict(torch.load("best_model.pt"))
    return model


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

# --- 6. Maak Datasets en Loaders ---
train_dataset = IncomeDataset(X_tr, y_tr)
val_dataset = IncomeDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

# --- 7. Initialiseer model ---
num_numeric_features = len(numeric_cols)               # 4 (year, age, prestg10, childs)
num_onehot_features = X_tr.shape[1] - len(numeric_cols) - 2  # totaal min numerics, educcat, occrecode
occrecode_cardinality = X_train["occrecode"].nunique() # aantal unieke beroepen

model = IncomeNN(
    num_numeric_features=num_numeric_features,
    num_onehot_features=num_onehot_features,
    occrecode_cardinality=occrecode_cardinality,
    embedding_dim=16,
    hidden_dims=[128, 64]
)

# --- 8. Train model ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = train_model(model, train_loader, val_loader, epochs=50, lr=1e-3, patience=5, device=device)

