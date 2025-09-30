# CRPS WAS 8400 HIER, GEBRUIK VAN VERSCHILLENDE DENSITIES
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scoringrules import crps_ensemble

# -------------------------
# MDN: Mixture Density Network (K=3)
# -------------------------
class IncomeMDN(nn.Module):
    def __init__(self, num_numeric_features, num_onehot_features,
                 occrecode_cardinality, embedding_dim=16,
                 hidden_dims=[128, 64], num_components=3):
        super().__init__()
        self.num_components = num_components
        self.occrecode_emb = nn.Embedding(occrecode_cardinality, embedding_dim)

        input_dim = num_numeric_features + num_onehot_features + 1 + embedding_dim
        layers = []
        dim_in = input_dim
        for dim_out in hidden_dims:
            layers.append(nn.Linear(dim_in, dim_out))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.15))
            dim_in = dim_out
        self.hidden = nn.Sequential(*layers)

        # Drie koppen: pi, mu, sigma
        self.pi_head = nn.Linear(dim_in, num_components)
        self.mu_head = nn.Linear(dim_in, num_components)
        self.sigma_head = nn.Linear(dim_in, num_components)

    def forward(self, numeric, onehot, educcat, occrecode):
        occ_emb = self.occrecode_emb(occrecode)
        x = torch.cat([numeric, onehot, educcat, occ_emb], dim=1)
        h = self.hidden(x)

        pi = F.softmax(self.pi_head(h), dim=1)     # (batch,K)
        mu = self.mu_head(h)                       # (batch,K)
        sigma = F.softplus(self.sigma_head(h)) + 1e-6  # (batch,K)
        return pi, mu, sigma


# -------------------------
# Loss: NLL for mixture of log-normals
# -------------------------
def mdn_log_normal_nll(y, pi, mu, sigma):
    """
    y: (batch,)
    pi, mu, sigma: (batch,K)
    """
    logy = torch.log(y + 1e-8).unsqueeze(1)  # (batch,1)
    sqrt_two_pi = torch.sqrt(torch.tensor(2.0 * torch.pi, device=y.device))

    coeff = 1.0 / (y.unsqueeze(1) * sigma * sqrt_two_pi)   # (batch,K)
    exponent = -0.5 * ((logy - mu) ** 2) / (sigma ** 2)    # (batch,K)
    component_pdf = coeff * torch.exp(exponent)            # (batch,K)

    weighted_pdf = pi * component_pdf
    pdf_sum = torch.sum(weighted_pdf, dim=1) + 1e-12       # (batch,)
    nll = -torch.log(pdf_sum)
    return nll.mean()


# -------------------------
# Training loop (update loss call)
# -------------------------
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
            numeric, onehot, educcat, occrecode, y = (
                numeric.to(device), onehot.to(device),
                educcat.to(device), occrecode.to(device), y.to(device)
            )
            optimizer.zero_grad()
            pi, mu, sigma = model(numeric, onehot, educcat, occrecode)
            loss = mdn_log_normal_nll(y, pi, mu, sigma)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * y.size(0)
        train_loss /= len(train_loader.dataset)

        # --- Validation ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for numeric, onehot, educcat, occrecode, y in val_loader:
                numeric, onehot, educcat, occrecode, y = (
                    numeric.to(device), onehot.to(device),
                    educcat.to(device), occrecode.to(device), y.to(device)
                )
                pi, mu, sigma = model(numeric, onehot, educcat, occrecode)
                loss = mdn_log_normal_nll(y, pi, mu, sigma)
                val_loss += loss.item() * y.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

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


# -------------------------
# Sampling from MDN
# -------------------------
def sample_from_mdn(pi, mu, sigma, n_samples=1000):
    """
    Sample n_samples per observation from mixture of log-normals
    pi, mu, sigma: (batch,K)
    return: (batch, n_samples)
    """
    batch_size, K = pi.shape
    cat = torch.distributions.Categorical(pi)
    components = cat.sample((n_samples,)).T  # (batch,n_samples)

    eps = torch.randn(batch_size, n_samples, device=pi.device)
    mu_sel = mu.gather(1, components)
    sigma_sel = sigma.gather(1, components)

    logy = mu_sel + sigma_sel * eps
    y_samples = torch.exp(logy)
    return y_samples

# --- Load data ---
X_train = pd.read_csv('X_trn.csv')
y_train = pd.read_csv('Y_trn.csv')
X_test = pd.read_csv('X_test.csv') # ONLY FOR THE FINAL PREDICTIONS, NOT USED IN TRAINING OR VALIDATION.
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


train_dataset = IncomeDataset(X_tr, y_tr)
val_dataset = IncomeDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)


# --- 7. Initialiseer model ---
num_numeric_features = len(numeric_cols)
num_onehot_features = X_tr.shape[1] - len(numeric_cols) - 2  # minus educcat + occrecode
occrecode_cardinality = X_train["occrecode"].nunique()

model = IncomeMDN(
    num_numeric_features=num_numeric_features,
    num_onehot_features=num_onehot_features,
    occrecode_cardinality=occrecode_cardinality,
    embedding_dim=16,
    hidden_dims=[128, 64],
    num_components=3
)


# --- 8. Train model ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = train_model(model, train_loader, val_loader, epochs=50, lr=1e-3, patience=5, device=device)


# --- 9. Bereken CRPS op de validatieset ---
val_samples = []
y_val_array = []

model.eval()
with torch.no_grad():
    for numeric, onehot, educcat, occrecode, y in val_loader:
        numeric, onehot, educcat, occrecode, y = (
            numeric.to(device), onehot.to(device),
            educcat.to(device), occrecode.to(device), y.to(device)
        )
        pi, mu, sigma = model(numeric, onehot, educcat, occrecode)
        samples = sample_from_mdn(pi, mu, sigma, n_samples=1000)  # (batch,1000)
        val_samples.append(samples.cpu())
        y_val_array.append(y.cpu())

val_pred = torch.cat(val_samples, dim=0).numpy()
y_val_array = torch.cat(y_val_array, dim=0).numpy()
crps_val = crps_ensemble(y_val_array, val_pred).mean().item()
print("Validation CRPS:", crps_val)


# --- 10. Preprocess test set ---
X_test_proc = X_test.copy()
X_test_proc["educcat"] = educcat_encoder.transform(X_test_proc[["educcat"]]).astype(int)
X_test_proc = pd.get_dummies(X_test_proc, columns=onehot_cols, drop_first=True)
X_test_proc = X_test_proc.reindex(columns=X_train.columns, fill_value=0)
X_test_proc["occrecode"] = X_test_proc["occrecode"].map(occrecode_mapping)
X_test_proc[numeric_cols] = scaler.transform(X_test_proc[numeric_cols])


# --- 11. Dataset & DataLoader voor testset ---
class IncomeTestDataset(Dataset):
    def __init__(self, df):
        self.numeric = torch.tensor(df[["year", "age", "prestg10", "childs"]].values, dtype=torch.float32)
        self.onehot = torch.tensor(df.drop(columns=["year","age","prestg10","childs","educcat","occrecode"]).values, dtype=torch.float32)
        self.educcat = torch.tensor(df["educcat"].values, dtype=torch.float32).unsqueeze(1)
        self.occrecode = torch.tensor(df["occrecode"].values, dtype=torch.long)
    def __len__(self):
        return len(self.numeric)
    def __getitem__(self, idx):
        return self.numeric[idx], self.onehot[idx], self.educcat[idx], self.occrecode[idx]

test_dataset = IncomeTestDataset(X_test_proc)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)


# --- 12. Monte Carlo samples per observatie (test predictions) ---
model.eval()
all_samples = []
with torch.no_grad():
    for numeric, onehot, educcat, occrecode in test_loader:
        numeric, onehot, educcat, occrecode = (
            numeric.to(device), onehot.to(device), educcat.to(device), occrecode.to(device)
        )
        pi, mu, sigma = model(numeric, onehot, educcat, occrecode)
        samples = sample_from_mdn(pi, mu, sigma, n_samples=1000)
        all_samples.append(samples.cpu())

y_pred = torch.cat(all_samples, dim=0).numpy()  # shape (n_obs, 1000)
print("Prediction matrix shape:", y_pred.shape)
np.save("predictions.npy", y_pred)
print("predictions.npy saved!")
