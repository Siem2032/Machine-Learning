# CRPS 8471, gestabiliseerde MDN (K=3) met regelmatige verbeteringen: stabiele log-sum-exp NLL, sigma-floor + penalty, entropy regularizer op π, gradient clipping, lagere learning rate en weight decay.
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from scoringrules import crps_ensemble

# ---------------------------
# Repro
# ---------------------------
RND = 42
random.seed(RND)
np.random.seed(RND)
torch.manual_seed(RND)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RND)

# ---------------------------
# Dataset classes
# ---------------------------
class IncomeDataset(Dataset):
    def __init__(self, df, y):
        # Assumes df is already preprocessed and columns aligned with training pipeline
        self.numeric = torch.tensor(df[["year", "age", "prestg10", "childs"]].values, dtype=torch.float32)
        # onehot = all columns except the numeric ones + educcat + occrecode
        self.onehot = torch.tensor(df.drop(columns=["year", "age", "prestg10", "childs", "educcat", "occrecode"]).values, dtype=torch.float32)
        self.educcat = torch.tensor(df["educcat"].values, dtype=torch.float32).unsqueeze(1)
        self.occrecode = torch.tensor(df["occrecode"].values, dtype=torch.long)
        self.y = torch.tensor(y.values.squeeze(), dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.numeric[idx], self.onehot[idx], self.educcat[idx], self.occrecode[idx], self.y[idx]

class IncomeTestDataset(Dataset):
    def __init__(self, df):
        self.numeric = torch.tensor(df[["year", "age", "prestg10", "childs"]].values, dtype=torch.float32)
        self.onehot = torch.tensor(df.drop(columns=["year", "age", "prestg10", "childs", "educcat", "occrecode"]).values, dtype=torch.float32)
        self.educcat = torch.tensor(df["educcat"].values, dtype=torch.float32).unsqueeze(1)
        self.occrecode = torch.tensor(df["occrecode"].values, dtype=torch.long)

    def __len__(self):
        return len(self.numeric)

    def __getitem__(self, idx):
        return self.numeric[idx], self.onehot[idx], self.educcat[idx], self.occrecode[idx]

# ---------------------------
# MDN Model (K components)
# ---------------------------
class IncomeMDN(nn.Module):
    def __init__(self, num_numeric_features, num_onehot_features,
                 occrecode_cardinality, embedding_dim=16,
                 hidden_dims=[128,64], num_components=3):
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

        self.pi_head = nn.Linear(dim_in, num_components)
        self.mu_head = nn.Linear(dim_in, num_components)
        self.sigma_head = nn.Linear(dim_in, num_components)

    def forward(self, numeric, onehot, educcat, occrecode):
        occ_emb = self.occrecode_emb(occrecode)
        x = torch.cat([numeric, onehot, educcat, occ_emb], dim=1)
        h = self.hidden(x)
        pi = F.softmax(self.pi_head(h), dim=1)
        mu = self.mu_head(h)
        sigma = F.softplus(self.sigma_head(h)) + 1e-6
        return pi, mu, sigma

# ---------------------------
# Stable MDN NLL + regs
# ---------------------------
def mdn_log_normal_nll_with_regs(y, pi, mu, sigma, sigma_min=1e-2, lambda_sigma=1e-3, lambda_entropy=1e-4):
    """
    Stable negative log-likelihood for mixture of log-normals + small regularizers.
    y: (batch,)
    pi, mu, sigma: (batch, K)
    """
    eps = 1e-12
    # log(y) in shape (batch,1)
    logy = torch.log(y + 1e-8).unsqueeze(1)  # (batch,1)

    # log component density of log-normal:
    # log p(y|k) = -log(y) - log(sigma) - 0.5*log(2pi) - 0.5 * ((logy - mu)^2 / sigma^2)
    const = torch.log(torch.tensor(2.0 * torch.pi, device=y.device, dtype=y.dtype))
    log_component = (
        - torch.log(y + 1e-8).unsqueeze(1)
        - torch.log(sigma + eps)
        - 0.5 * const
        - 0.5 * ((logy - mu) ** 2) / (sigma ** 2 + eps)
    )  # (batch, K)

    log_pi = torch.log(pi + eps)  # (batch,K)
    log_weighted = log_pi + log_component  # (batch,K)

    # log-sum-exp for numerical stability
    max_log, _ = torch.max(log_weighted, dim=1, keepdim=True)  # (batch,1)
    lse = max_log + torch.log(torch.sum(torch.exp(log_weighted - max_log), dim=1, keepdim=True) + eps)  # (batch,1)
    log_pdf = lse.squeeze(1)  # (batch,)

    nll = -log_pdf.mean()

    # Regularizers
    sigma_pen = torch.relu(sigma_min - sigma).mean()  # penalize very small sigma
    entropy = -torch.sum(pi * torch.log(pi + eps), dim=1).mean()  # average entropy

    loss = nll + lambda_sigma * sigma_pen - lambda_entropy * entropy
    return loss

# ---------------------------
# Sampling from MDN
# ---------------------------
def sample_from_mdn(pi, mu, sigma, n_samples=1000):
    """
    Return shape (batch, n_samples) with positive y samples (log-normal mixture).
    """
    batch_size, K = pi.shape
    # sample component indices per draw
    # cat.sample((n_samples,)) -> shape (n_samples, batch).T -> (batch, n_samples)
    cat = torch.distributions.Categorical(pi)
    components = cat.sample((n_samples,)).T  # long tensor (batch, n_samples)

    # eps for normals
    eps = torch.randn(batch_size, n_samples, device=pi.device)
    # gather mu and sigma per-draw
    mu_sel = mu.gather(1, components)       # (batch, n_samples)
    sigma_sel = sigma.gather(1, components) # (batch, n_samples)
    logy = mu_sel + sigma_sel * eps
    y_samples = torch.exp(logy)
    return y_samples  # (batch, n_samples)

# ---------------------------
# Helper: train loop with grad clipping + scheduler
# ---------------------------
def train_model(model, train_loader, val_loader, epochs=200, lr=1e-4, patience=15, device="cpu",
                weight_decay=1e-6, max_grad_norm=5.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    best_val = float('inf')
    patience_counter = 0
    model.to(device)

    for epoch in range(1, epochs+1):
        model.train()
        train_loss_acc = 0.0
        n_train = 0
        for numeric, onehot, educcat, occrecode, y in train_loader:
            numeric = numeric.to(device); onehot = onehot.to(device); educcat = educcat.to(device)
            occrecode = occrecode.to(device); y = y.to(device)

            optimizer.zero_grad()
            pi, mu, sigma = model(numeric, onehot, educcat, occrecode)
            loss = mdn_log_normal_nll_with_regs(y, pi, mu, sigma)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            bs = y.size(0)
            train_loss_acc += loss.item() * bs
            n_train += bs

        train_loss = train_loss_acc / n_train

        # validation
        model.eval()
        val_loss_acc = 0.0
        n_val = 0
        with torch.no_grad():
            for numeric, onehot, educcat, occrecode, y in val_loader:
                numeric = numeric.to(device); onehot = onehot.to(device); educcat = educcat.to(device)
                occrecode = occrecode.to(device); y = y.to(device)
                pi, mu, sigma = model(numeric, onehot, educcat, occrecode)
                loss = mdn_log_normal_nll_with_regs(y, pi, mu, sigma)
                bs = y.size(0)
                val_loss_acc += loss.item() * bs
                n_val += bs
        val_loss = val_loss_acc / n_val

        scheduler.step(val_loss)

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.1e}")

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_mdn_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(torch.load("best_mdn_model.pt"))
    return model

# ---------------------------
# Main pipeline
# ---------------------------
def main():
    # Paths - adjust if needed
    Xtr_path = "X_trn.csv"
    ytr_path = "Y_trn.csv"
    Xtest_path = "X_test.csv"

    # Load
    X = pd.read_csv(Xtr_path)
    y = pd.read_csv(ytr_path)
    X_test = pd.read_csv(Xtest_path)

    # 1) educcat ordinal encoder (fit on training)
    educcat_categories = [["Less Than High School", "High School", "Junior College", "Bachelor", "Graduate"]]
    educcat_encoder = OrdinalEncoder(categories=educcat_categories)
    X["educcat"] = educcat_encoder.fit_transform(X[["educcat"]]).astype(int)

    # 2) one-hot small categoricals on training
    onehot_cols = ["gender", "maritalcat", "wrkstat"]
    X = pd.get_dummies(X, columns=onehot_cols, drop_first=True)
    train_columns = X.columns  # save column order for test reindex

    # 3) occrecode mapping (train-based). Add an explicit 'unknown' index at the end.
    unique_occs = list(X["occrecode"].unique())
    occrecode_mapping = {cat: idx for idx, cat in enumerate(unique_occs)}
    unknown_occ_idx = len(occrecode_mapping)  # index for unseen occs in test
    occrecode_cardinality = len(occrecode_mapping) + 1  # +1 for unknown
    X["occrecode"] = X["occrecode"].map(occrecode_mapping)

    # 4) standardize numerics
    numeric_cols = ["year", "age", "prestg10", "childs"]
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # 5) split train/val
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=RND)

    # Build datasets and loaders
    train_dataset = IncomeDataset(X_tr, y_tr)
    val_dataset = IncomeDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # compute num_onehot_features properly:
    num_numeric_features = len(numeric_cols)
    num_onehot_features = X_tr.shape[1] - len(numeric_cols) - 2  # minus educcat and occrecode

    # initialize model
    model = IncomeMDN(
        num_numeric_features=num_numeric_features,
        num_onehot_features=num_onehot_features,
        occrecode_cardinality=occrecode_cardinality,
        embedding_dim=32,
        hidden_dims=[256, 128],
        num_components=3
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train_model(model, train_loader, val_loader, epochs=200, lr=1e-4, patience=15, device=device)

    # Validation CRPS (sample-based)
    model.eval()
    val_samples = []
    y_val_array = []
    with torch.no_grad():
        for numeric, onehot, educcat, occrecode, y_batch in val_loader:
            numeric = numeric.to(device); onehot = onehot.to(device); educcat = educcat.to(device); occrecode = occrecode.to(device)
            pi, mu, sigma = model(numeric, onehot, educcat, occrecode)
            samples = sample_from_mdn(pi, mu, sigma, n_samples=1000)  # (batch, 1000)
            val_samples.append(samples.cpu())
            y_val_array.append(y_batch.cpu())
    val_pred = torch.cat(val_samples, dim=0).numpy()
    y_val_array = torch.cat(y_val_array, dim=0).numpy().squeeze()
    crps_val = crps_ensemble(y_val_array, val_pred).mean().item()
    print("Validation CRPS:", crps_val)

    # ---------------------------
    # Preprocess test set exactly same way
    # ---------------------------
    X_test_proc = X_test.copy()
    # educcat
    X_test_proc["educcat"] = educcat_encoder.transform(X_test_proc[["educcat"]]).astype(int)
    # one-hot and align columns
    X_test_proc = pd.get_dummies(X_test_proc, columns=onehot_cols, drop_first=True)
    X_test_proc = X_test_proc.reindex(columns=train_columns, fill_value=0)
    # occrecode mapping with unknown handling
    X_test_proc["occrecode"] = X_test_proc["occrecode"].map(lambda v: occrecode_mapping.get(v, unknown_occ_idx))
    # scale numeric columns
    X_test_proc[numeric_cols] = scaler.transform(X_test_proc[numeric_cols])

    # test dataset and loader
    test_dataset = IncomeTestDataset(X_test_proc)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    # produce samples for test set and save predictions.npy
    all_samples = []
    model.eval()
    with torch.no_grad():
        for numeric, onehot, educcat, occrecode in test_loader:
            numeric = numeric.to(device); onehot = onehot.to(device); educcat = educcat.to(device); occrecode = occrecode.to(device)
            pi, mu, sigma = model(numeric, onehot, educcat, occrecode)
            samples = sample_from_mdn(pi, mu, sigma, n_samples=1000)
            all_samples.append(samples.cpu())

    y_pred = torch.cat(all_samples, dim=0).numpy()  # (n_obs, 1000)
    print("Prediction matrix shape:", y_pred.shape)
    np.save("predictions.npy", y_pred)
    print("Saved predictions.npy (shape {})".format(y_pred.shape))

if __name__ == "__main__":
    main()
