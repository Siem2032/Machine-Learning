# CRPS is 7966 in deze code Best hyperparams: {'n_components': 3, 'n_layers': 1, 'hidden_size': 64, 'lr': 0.006161049539380964, 'weight_decay': 0.0002463768595899747, 'dropout': 0.36874969400924673, 'occre_emb_dim': 16, 'batch_size': 128, 'min_sigma': 0.0017247957710046008, 'use_entropy_pen': 0}

import os
import math
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

import optuna

# ----------------------------
# Reproducibility / device
# ----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ----------------------------
# CRPS for ensemble samples
# samples: (n_obs, m), y: (n_obs,)
# ----------------------------
def crps_ensemble(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    n, m = samples.shape
    # term1: E|X-y|
    term1 = np.mean(np.abs(samples - y.reshape(-1,1)), axis=1)
    # term2: 0.5 E|X-X'|
    term2 = np.zeros(n, dtype=float)
    # compute pairwise diffs per observation
    for i in range(n):
        s = samples[i]
        diffs = np.abs(s.reshape(-1,1) - s.reshape(1,-1))
        term2[i] = 0.5 * diffs.mean()
    return term1 - term2

# ----------------------------
# MDN model
# ----------------------------
class MDN(nn.Module):
    def __init__(self, n_features:int, hidden_sizes:List[int], n_components:int,
                 occrecode_dim:int=None, occrecode_emb_dim:int=16, dropout:float=0.2):
        super().__init__()
        self.n_components = n_components
        self.occrecode_dim = occrecode_dim
        self.occrecode_emb_dim = occrecode_emb_dim if occrecode_dim is not None else 0

        input_dim = n_features
        # trunk
        layers = []
        in_dim = input_dim + self.occrecode_emb_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        self.trunk = nn.Sequential(*layers)
        # heads
        self.pi_layer = nn.Linear(in_dim, n_components)
        self.mu_layer = nn.Linear(in_dim, n_components)
        self.log_sigma_layer = nn.Linear(in_dim, n_components)
        # embedding
        if occrecode_dim is not None:
            self.occre_emb = nn.Embedding(occrecode_dim, self.occrecode_emb_dim)
        else:
            self.occre_emb = None

    def forward(self, x_cont: torch.Tensor, occre_idx: torch.Tensor = None):
        if self.occre_emb is not None and occre_idx is not None:
            e = self.occre_emb(occre_idx)
            x = torch.cat([x_cont, e], dim=1)
        else:
            x = x_cont
        h = self.trunk(x)
        logits = self.pi_layer(h)
        mu = self.mu_layer(h)
        log_sigma = self.log_sigma_layer(h)
        return logits, mu, log_sigma

# ----------------------------
# MDN negative log-likelihood (stable)
# ----------------------------
def mdn_negloglikelihood(logits: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor,
                         y: torch.Tensor, min_sigma: float = 1e-4) -> (torch.Tensor, torch.Tensor):
    # logits, mu, log_sigma: (batch, K)
    # y: (batch,) or (batch,1)
    y = y.view(-1,1)  # (batch,1)
    log_pi = logits - torch.logsumexp(logits, dim=1, keepdim=True)  # (batch,K)
    sigma = F.softplus(log_sigma) + min_sigma  # (batch,K)
    z = (y - mu) / sigma
    comp_logprob = -0.5 * (z**2 + 2.0*torch.log(sigma) + math.log(2*math.pi))  # (batch,K)
    log_weighted = log_pi + comp_logprob
    lls = torch.logsumexp(log_weighted, dim=1)  # (batch,)
    nll = - torch.mean(lls)
    return nll, lls

# ----------------------------
# Sampling from MDN params (returns numpy array)
# logits, mu, log_sigma are torch tensors (batch, K)
# ----------------------------
def mdn_sample_from_params(logits: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor,
                           n_samples: int = 1000, min_sigma: float = 1e-4) -> np.ndarray:
    # convert to numpy safely
    logits_np = logits.detach().cpu().numpy()
    mu_np = mu.detach().cpu().numpy()
    sigma_np = F.softplus(log_sigma).detach().cpu().numpy() + min_sigma
    n_obs, K = logits_np.shape
    # softmax
    pi = np.exp(logits_np - logits_np.max(axis=1, keepdims=True))
    pi = pi / pi.sum(axis=1, keepdims=True)
    samples = np.zeros((n_obs, n_samples), dtype=float)
    for i in range(n_obs):
        comps = np.random.choice(K, size=n_samples, p=pi[i])
        eps = np.random.randn(n_samples)
        samples[i, :] = mu_np[i, comps] + sigma_np[i, comps] * eps
    return samples

# ----------------------------
# Preprocess / prepare data
# ----------------------------
def prepare_data(X_train_path='X_trn.csv', y_train_path='Y_trn.csv', X_test_path='X_test.csv'):
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).squeeze()
    X_test = pd.read_csv(X_test_path)

    # Target: log1p transform (train on log-scale)
    y_train = np.log1p(y_train)

    # Feature engineering: prestg10 log1p if exists
    if "prestg10" in X_train.columns:
        X_train["prestg10"] = np.log1p(X_train["prestg10"])
        if "prestg10" in X_test.columns:
            X_test["prestg10"] = np.log1p(X_test["prestg10"])

    # Ordinal encode educcat
    educcat_categories = [["Less Than High School", "High School", "Junior College", "Bachelor", "Graduate"]]
    enc = OrdinalEncoder(categories=educcat_categories)
    X_train["educcat"] = enc.fit_transform(X_train[["educcat"]]).astype(int)
    try:
        X_test["educcat"] = enc.transform(X_test[["educcat"]]).astype(int)
    except Exception:
        pass

    # One-hot small categoricals
    onehot_cols = ["gender", "maritalcat", "wrkstat"]
    X_train = pd.get_dummies(X_train, columns=onehot_cols, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=onehot_cols, drop_first=True)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # occrecode mapping -> integers for embedding
    occrecode_mapping = {cat: idx for idx, cat in enumerate(X_train["occrecode"].unique())}
    X_train["occrecode"] = X_train["occrecode"].map(occrecode_mapping)
    X_test["occrecode"] = X_test["occrecode"].map(lambda v: occrecode_mapping.get(v, max(occrecode_mapping.values())+1))

    return X_train, y_train, X_test, occrecode_mapping

# ----------------------------
# Train fold
# ----------------------------
def train_mdn_fold(X: pd.DataFrame, y: pd.Series, occrecode_mapping: Dict[Any,int],
                   train_idx: np.ndarray, val_idx: np.ndarray, config: Dict[str,Any],
                   device: torch.device) -> (MDN, StandardScaler, float):
    occ_col = 'occrecode'
    cont_cols = [c for c in X.columns if c != occ_col]
    Xc = X[cont_cols].values.astype(np.float32)
    occ_idx = X[occ_col].values.astype(np.int64)

    # scaler fit on train only
    scaler = StandardScaler()
    scaler.fit(Xc[train_idx])
    Xc_scaled = scaler.transform(Xc)

    # tensors
    X_train_t = torch.from_numpy(Xc_scaled[train_idx])
    occ_train_t = torch.from_numpy(occ_idx[train_idx])
    y_train_t = torch.from_numpy(y.values[train_idx].astype(np.float32))
    X_val_t = torch.from_numpy(Xc_scaled[val_idx])
    occ_val_t = torch.from_numpy(occ_idx[val_idx])
    y_val_t = torch.from_numpy(y.values[val_idx].astype(np.float32))

    train_ds = TensorDataset(X_train_t, occ_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, occ_val_t, y_val_t)

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False)

    n_features = Xc.shape[1]
    occre_dim = max(occrecode_mapping.values()) + 2

    model = MDN(n_features=n_features,
                hidden_sizes=config['hidden_sizes'],
                n_components=config['n_components'],
                occrecode_dim=occre_dim,
                occrecode_emb_dim=config['occre_emb_dim'],
                dropout=config['dropout']).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0
    patience = config.get('patience', 20)

    # init
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    for epoch in range(config['epochs']):
        model.train()
        for xb, occb, yb in train_loader:
            xb = xb.to(device)
            occb = occb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits, mu, log_sigma = model(xb, occb)
            nll, _ = mdn_negloglikelihood(logits, mu, log_sigma, yb, min_sigma=config['min_sigma'])
            # entropy penalty to avoid collapse (optional)
            log_pi = logits - torch.logsumexp(logits, dim=1, keepdim=True)
            entropy = -torch.sum(torch.exp(log_pi) * log_pi, dim=1).mean()
            entropy_pen = config.get('entropy_penalty', 0.0) * (-entropy)
            loss = nll + entropy_pen
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        # validation
        model.eval()
        with torch.no_grad():
            val_lls = []
            for xb, occb, yb in val_loader:
                xb = xb.to(device)
                occb = occb.to(device)
                yb = yb.to(device)
                logits_v, mu_v, log_sigma_v = model(xb, occb)   # <-- correct forward per batch
                nll_v, _ = mdn_negloglikelihood(logits_v, mu_v, log_sigma_v, yb, min_sigma=config['min_sigma'])
                val_lls.append(nll_v.item())
            val_loss = float(np.mean(val_lls))

        if val_loss < best_val_loss - 1e-9:
            best_val_loss = val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, scaler, best_val_loss

# ----------------------------
# Optuna objective (uses device passed)
# ----------------------------
def objective_optuna(trial, X, y, occrecode_mapping, groups, device: torch.device):
    n_components = trial.suggest_categorical("n_components", [3,5,8,12,16,20])
    n_layers = trial.suggest_int("n_layers", 1, 3)
    hidden_size = trial.suggest_categorical("hidden_size", [64,128,256])
    hidden_sizes = [hidden_size] * n_layers
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    oc_emb_dim = trial.suggest_categorical("occre_emb_dim", [8,16,32])
    batch_size = trial.suggest_categorical("batch_size", [64,128,256])
    min_sigma = trial.suggest_loguniform("min_sigma", 1e-5, 5e-3)
    use_entropy_pen = trial.suggest_int("use_entropy_pen", 0, 1)
    entropy_penalty = trial.suggest_loguniform("entropy_penalty", 1e-6, 1e-2) if use_entropy_pen else 0.0

    config = {
        'n_components': n_components,
        'hidden_sizes': hidden_sizes,
        'lr': lr,
        'weight_decay': weight_decay,
        'dropout': dropout,
        'occre_emb_dim': oc_emb_dim,
        'batch_size': batch_size,
        'epochs': 60,      # short for HPO
        'patience': 12,
        'min_sigma': min_sigma,
        'entropy_penalty': entropy_penalty
    }

    gkf = GroupKFold(n_splits=3)
    val_scores = []
    for train_idx, val_idx in gkf.split(X, y, groups):
        model, scaler, val_loss = train_mdn_fold(X, y, occrecode_mapping, train_idx, val_idx, config, device)
        val_scores.append(val_loss)
        trial.report(np.mean(val_scores), step=len(val_scores))
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(val_scores))

# ----------------------------
# Full pipeline
# ----------------------------
def run_full_pipeline(X_train_path='X_trn.csv', y_train_path='Y_trn.csv', X_test_path='X_test.csv',
                      out_predictions='predictions.npy', device_str: str = 'cuda'):
    device = torch.device(device_str if torch.cuda.is_available() and device_str=='cuda' else 'cpu')
    X, y_log, X_test, occrecode_mapping = prepare_data(X_train_path, y_train_path, X_test_path)

    # groups for GroupKFold (assumes 'year' exists)
    if 'year' not in X.columns:
        raise RuntimeError("Column 'year' must exist for GroupKFold splitting.")
    groups = X['year'].values
    # remove year from features
    X = X.drop(columns=['year'])
    X_test = X_test.drop(columns=['year']) if 'year' in X_test.columns else X_test

    # Optuna study
    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.HyperbandPruner())
    n_trials = 12   # keep small for quick runs; raise for final runs
    study.optimize(lambda trial: objective_optuna(trial, X, y_log, occrecode_mapping, groups, device), n_trials=n_trials)

    best = study.best_trial.params
    print("Best hyperparams:", best)

    # Build final config from best
    config = {
        'n_components': best.get('n_components', 8),
        'hidden_sizes': [best.get('hidden_size', 128)] * best.get('n_layers', 2),
        'lr': best.get('lr', 1e-3),
        'weight_decay': best.get('weight_decay', 1e-4),
        'dropout': best.get('dropout', 0.2),
        'occre_emb_dim': best.get('occre_emb_dim', 16),
        'batch_size': best.get('batch_size', 128),
        'epochs': 200,
        'patience': 25,
        'min_sigma': best.get('min_sigma', 1e-3),
        'entropy_penalty': best.get('entropy_penalty', 0.0)
    }

    # Train ensemble (GroupKFold)
    K = 5
    gkf = GroupKFold(n_splits=K)
    fold_models = []
    scalers = []
    val_losses = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_log, groups)):
        print(f"Training fold {fold+1}/{K} ...")
        model, scaler, val_loss = train_mdn_fold(X, y_log, occrecode_mapping, train_idx, val_idx, config, device)
        fold_models.append(model)
        scalers.append(scaler)
        val_losses.append(val_loss)
        # save fold if desired
        torch.save(model.state_dict(), f'mdn_fold_{fold}.pt')
    print("Fold val losses:", val_losses, "mean:", np.mean(val_losses))

    # Prepare test features
    occ_col = 'occrecode'
    cont_cols = [c for c in X_test.columns if c != occ_col]
    Xc_test = X_test[cont_cols].values.astype(np.float32)
    occ_idx_test = X_test[occ_col].values.astype(np.int64)

    # MC sampling from ensemble -> aggregated predictions (on original scale)
    N_MC = 1000
    n_test = X_test.shape[0]
    per_model = max(1, N_MC // len(fold_models))
    extra = N_MC - per_model * len(fold_models)
    all_samples = np.zeros((n_test, 0), dtype=float)

    for i, (model, scaler) in enumerate(zip(fold_models, scalers)):
        model.eval()
        Xc_scaled = scaler.transform(Xc_test)
        xb = torch.from_numpy(Xc_scaled).to(device)
        occb = torch.from_numpy(occ_idx_test).to(device)
        with torch.no_grad():
            logits, mu, log_sigma = model(xb, occb)
        m = per_model + (1 if i < extra else 0)
        s = mdn_sample_from_params(logits, mu, log_sigma, n_samples=m, min_sigma=config['min_sigma'])
        # back-transform from log1p -> original scale
        s = np.expm1(s)
        all_samples = np.concatenate([all_samples, s], axis=1)

    # ensure exactly N_MC columns
    if all_samples.shape[1] != N_MC:
        if all_samples.shape[1] > N_MC:
            idxs = np.random.choice(all_samples.shape[1], N_MC, replace=False)
            all_samples = all_samples[:, idxs]
        else:
            need = N_MC - all_samples.shape[1]
            idxs = np.random.choice(all_samples.shape[1], need, replace=True)
            pad = all_samples[:, idxs]
            all_samples = np.concatenate([all_samples, pad], axis=1)

    np.save(out_predictions, all_samples.astype(np.float32))
    print(f"Saved predictions.npy with shape {all_samples.shape}")

    # --------------------------
    # OOB CRPS on training set (back-transform y)
    # --------------------------
    print("Computing OOB CRPS (OOB predictions per fold)...")
    n_train = X.shape[0]
    N_MC_oob = 500
    oob_samples = np.zeros((n_train, N_MC_oob), dtype=float)
    got = np.zeros(n_train, dtype=bool)

    Xc_all = X[[c for c in X.columns if c != occ_col]].values.astype(np.float32)
    occ_idx_all = X[occ_col].values.astype(np.int64)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_log, groups)):
        model = fold_models[fold]
        scaler = scalers[fold]
        model.eval()
        Xc_val = scaler.transform(Xc_all[val_idx])
        xb = torch.from_numpy(Xc_val).to(device)
        occb = torch.from_numpy(occ_idx_all[val_idx]).to(device)
        with torch.no_grad():
            logits, mu, log_sigma = model(xb, occb)
        s = mdn_sample_from_params(logits, mu, log_sigma, n_samples=N_MC_oob, min_sigma=config['min_sigma'])
        s = np.expm1(s)  # to original scale
        oob_samples[val_idx] = s
        got[val_idx] = True

    assert got.all(), "Some training rows did not receive OOB predictions!"
    y_orig = np.expm1(y_log.values)  # back-transform targets
    crps_scores = crps_ensemble(oob_samples, y_orig)
    mean_crps = crps_scores.mean()
    print(f"Mean OOB CRPS (CV-estimate): {mean_crps:.5f}")

    return all_samples, mean_crps

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    X_trn = "X_trn.csv"
    Y_trn = "Y_trn.csv"
    X_tst = "X_test.csv"
    device_choice = 'cuda' if torch.cuda.is_available() else 'cpu'
    preds, mean_crps = run_full_pipeline(X_trn, Y_trn, X_tst, out_predictions='predictions.npy', device_str=device_choice)
    print("Done. Mean OOB CRPS:", mean_crps)
