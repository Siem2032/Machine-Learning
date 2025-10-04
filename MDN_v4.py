# MDN_v4.py - Mean OOB CRPS: 7862.05068 Best hyperparams: {'n_components': 12, 'n_layers': 2, 'hidden_size': 128, 'lr': 0.004883899673992301, 'weight_decay': 0.00020133605295952412, 'dropout': 0.20887141979135015, 'occre_emb_dim': 16, 'batch_size': 128, 'min_sigma': 5.223174782636258e-05, 'entropy_penalty': 1.095655226424608e-06}

import os
import math
import random
import time
from typing import List, Dict, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

import optuna

# ----------------------------
# Repro / device
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
# CRPS helper
# ----------------------------
def crps_ensemble(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    # samples: (n_obs, m), y: (n_obs,)
    n, m = samples.shape
    term1 = np.mean(np.abs(samples - y.reshape(-1,1)), axis=1)
    term2 = np.zeros(n, dtype=float)
    for i in range(n):
        s = samples[i]
        diffs = np.abs(s.reshape(-1,1) - s.reshape(1,-1))
        term2[i] = 0.5 * diffs.mean()
    return term1 - term2

# ----------------------------
# MDN model (with BatchNorm)
# ----------------------------
class MDN(nn.Module):
    def __init__(self, n_features:int, hidden_sizes:List[int], n_components:int,
                 occrecode_dim:int=None, occrecode_emb_dim:int=16, dropout:float=0.2, use_batchnorm:bool=True):
        super().__init__()
        self.n_components = n_components
        self.occrecode_dim = occrecode_dim
        self.occrecode_emb_dim = occrecode_emb_dim if occrecode_dim is not None else 0
        input_dim = n_features

        layers = []
        in_dim = input_dim + self.occrecode_emb_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        self.trunk = nn.Sequential(*layers)

        self.pi_layer = nn.Linear(in_dim, n_components)
        self.mu_layer = nn.Linear(in_dim, n_components)
        self.log_sigma_layer = nn.Linear(in_dim, n_components)

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
# NLL (numerically stable)
# ----------------------------
def mdn_negloglikelihood(logits: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor,
                         y: torch.Tensor, min_sigma: float = 1e-4) -> (torch.Tensor, torch.Tensor):
    y = y.view(-1,1)
    log_pi = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    # use scaled softplus to avoid extremely large sigma; min_sigma floor added
    sigma = F.softplus(log_sigma) * 1.0 + min_sigma
    z = (y - mu) / sigma
    comp_logprob = -0.5 * (z**2 + 2.0*torch.log(sigma) + math.log(2*math.pi))
    log_weighted = log_pi + comp_logprob
    lls = torch.logsumexp(log_weighted, dim=1)
    nll = - torch.mean(lls)
    return nll, lls

# ----------------------------
# sampling
# ----------------------------
def mdn_sample_from_params(logits: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor,
                           n_samples: int = 1000, min_sigma: float = 1e-4) -> np.ndarray:
    logits_np = logits.detach().cpu().numpy()
    mu_np = mu.detach().cpu().numpy()
    sigma_np = F.softplus(log_sigma).detach().cpu().numpy() + min_sigma
    n_obs, K = logits_np.shape
    pi = np.exp(logits_np - logits_np.max(axis=1, keepdims=True))
    pi = pi / pi.sum(axis=1, keepdims=True)
    samples = np.zeros((n_obs, n_samples), dtype=float)
    for i in range(n_obs):
        comps = np.random.choice(K, size=n_samples, p=pi[i])
        eps = np.random.randn(n_samples)
        samples[i, :] = mu_np[i, comps] + sigma_np[i, comps] * eps
    return samples

# ----------------------------
# prepare_data (with extra feature engineering)
# ----------------------------
def prepare_data(X_train_path='X_trn.csv', y_train_path='Y_trn.csv', X_test_path='X_test.csv'):
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).squeeze()
    X_test = pd.read_csv(X_test_path)

    # target: log1p
    y_log = np.log1p(y_train)

    # feature engineering: log1p on skewed numeric columns (detect automatically)
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    # exclude columns that shouldn't be transformed
    exclude = {'year'}  # keep year for grouping initially
    numeric_cols = [c for c in numeric_cols if c not in exclude]
    # compute skewness and transform columns with skew > 1.0 (heuristic)
    skewed = X_train[numeric_cols].skew().abs()
    for col, skew_val in skewed.items():
        if skew_val > 1.0:
            # avoid transforming occrecode (categorical integers)
            if col in ['occrecode']:
                continue
            X_train[col] = np.log1p(X_train[col])
            if col in X_test.columns:
                X_test[col] = np.log1p(X_test[col])

    # ordinal encode education FIRST
    educcat_categories = [["Less Than High School", "High School", "Junior College", "Bachelor", "Graduate"]]
    enc = OrdinalEncoder(categories=educcat_categories)
    X_train["educcat"] = enc.fit_transform(X_train[["educcat"]]).astype(int)
    try:
        X_test["educcat"] = enc.transform(X_test[["educcat"]]).astype(int)
    except Exception:
        pass

    # now safe to build interactions
    if "prestg10" in X_train.columns:
        X_train["prestg10_sq"] = X_train["prestg10"] ** 2
        if "prestg10" in X_test.columns:
            X_test["prestg10_sq"] = X_test["prestg10"] ** 2
    if "educcat" in X_train.columns and "prestg10" in X_train.columns:
        X_train["edu_prestg"] = X_train["educcat"] * X_train["prestg10"]
        if set(["educcat","prestg10"]).issubset(X_test.columns):
            X_test["edu_prestg"] = X_test["educcat"] * X_test["prestg10"]

    try:
        X_test["educcat"] = enc.transform(X_test[["educcat"]]).astype(int)
    except Exception:
        pass

    # one-hot small categoricals
    onehot_cols = ["gender", "maritalcat", "wrkstat"]
    X_train = pd.get_dummies(X_train, columns=onehot_cols, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=onehot_cols, drop_first=True)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # occrecode mapping for embedding
    occrecode_mapping = {cat: idx for idx, cat in enumerate(X_train["occrecode"].unique())}
    X_train["occrecode"] = X_train["occrecode"].map(occrecode_mapping)
    X_test["occrecode"] = X_test["occrecode"].map(lambda v: occrecode_mapping.get(v, max(occrecode_mapping.values())+1))

    return X_train, y_log, X_test, occrecode_mapping

# ----------------------------
# Train function for a fold
# ----------------------------
def train_mdn_fold(X, y_log, occrecode_mapping, train_idx, val_idx, config, device):
    occ_col = 'occrecode'
    cont_cols = [c for c in X.columns if c != occ_col]
    Xc = X[cont_cols].values.astype(np.float32)
    occ_idx = X[occ_col].values.astype(np.int64)

    scaler = StandardScaler()
    scaler.fit(Xc[train_idx])
    Xc_scaled = scaler.transform(Xc)

    X_train_t = torch.from_numpy(Xc_scaled[train_idx])
    occ_train_t = torch.from_numpy(occ_idx[train_idx])
    y_train_t = torch.from_numpy(y_log.values[train_idx].astype(np.float32))
    X_val_t = torch.from_numpy(Xc_scaled[val_idx])
    occ_val_t = torch.from_numpy(occ_idx[val_idx])
    y_val_t = torch.from_numpy(y_log.values[val_idx].astype(np.float32))

    train_ds = TensorDataset(X_train_t, occ_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, occ_val_t, y_val_t)

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, pin_memory=(device.type=='cuda'))
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, pin_memory=(device.type=='cuda'))

    n_features = Xc.shape[1]
    occre_dim = max(occrecode_mapping.values()) + 2

    model = MDN(n_features=n_features,
                hidden_sizes=config['hidden_sizes'],
                n_components=config['n_components'],
                occrecode_dim=occre_dim,
                occrecode_emb_dim=config['occre_emb_dim'],
                dropout=config['dropout'],
                use_batchnorm=config.get('use_batchnorm', True)).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0
    patience = config.get('patience', 20)

    # init weights
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
                logits_v, mu_v, log_sigma_v = model(xb, occb)
                nll_v, _ = mdn_negloglikelihood(logits_v, mu_v, log_sigma_v, yb, min_sigma=config['min_sigma'])
                val_lls.append(nll_v.item())
            val_loss = float(np.mean(val_lls))

        scheduler.step(val_loss)
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
# Optuna objective
# ----------------------------
def objective_optuna(trial, X, y_log, occrecode_mapping, groups, device):
    n_components = trial.suggest_categorical("n_components", [5,8,12,16,20])
    n_layers = trial.suggest_int("n_layers", 1, 3)
    hidden_size = trial.suggest_categorical("hidden_size", [64,128,256])
    hidden_sizes = [hidden_size] * n_layers
    lr = trial.suggest_loguniform("lr", 1e-4, 5e-3)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)
    dropout = trial.suggest_float("dropout", 0.05, 0.35)
    oc_emb_dim = trial.suggest_categorical("occre_emb_dim", [8,16,32])
    batch_size = trial.suggest_categorical("batch_size", [64,128,256])
    min_sigma = trial.suggest_loguniform("min_sigma", 1e-6, 5e-4)
    # force entropy search to be tried
    entropy_penalty = trial.suggest_loguniform("entropy_penalty", 1e-6, 1e-2)

    config = {
        'n_components': n_components,
        'hidden_sizes': hidden_sizes,
        'lr': lr,
        'weight_decay': weight_decay,
        'dropout': dropout,
        'occre_emb_dim': oc_emb_dim,
        'batch_size': batch_size,
        'epochs': 60,
        'patience': 12,
        'min_sigma': min_sigma,
        'entropy_penalty': entropy_penalty,
        'occre_emb_dim': oc_emb_dim,
        'occre_emb_dim': oc_emb_dim,
    }

    gkf = GroupKFold(n_splits=3)
    scores = []
    for train_idx, val_idx in gkf.split(X, y_log, groups):
        model, scaler, val_loss = train_mdn_fold(X, y_log, occrecode_mapping, train_idx, val_idx, config, device)
        scores.append(val_loss)
        trial.report(np.mean(scores), step=len(scores))
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(scores))

# ----------------------------
# Full pipeline
# ----------------------------
def run_full_pipeline(X_train_path='X_trn.csv', y_train_path='Y_trn.csv', X_test_path='X_test.csv',
                      out_predictions='predictions.npy', device_str: str = 'cuda', n_trials: int = 40):
    device = torch.device(device_str if torch.cuda.is_available() and device_str=='cuda' else 'cpu')
    print("Using device:", device)

    X, y_log, X_test, occrecode_mapping = prepare_data(X_train_path, y_train_path, X_test_path)

    if 'year' not in X.columns:
        raise RuntimeError("Column 'year' must exist for GroupKFold grouping.")
    groups = X['year'].values
    X = X.drop(columns=['year'])
    if 'year' in X_test.columns:
        X_test = X_test.drop(columns=['year'])

    # Optuna study
    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.HyperbandPruner())
    print(f"Starting Optuna HPO (n_trials={n_trials}) ...")
    study.optimize(lambda trial: objective_optuna(trial, X, y_log, occrecode_mapping, groups, device), n_trials=n_trials)
    best = study.best_trial.params
    print("Best hyperparams:", best)

    # final config
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
        'min_sigma': best.get('min_sigma', 1e-4),
        'entropy_penalty': best.get('entropy_penalty', 1e-4),
        'use_batchnorm': True
    }

    # Train ensemble K-fold
    K = 5
    gkf = GroupKFold(n_splits=K)
    fold_models = []
    scalers = []
    val_losses = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_log, groups)):
        print(f"Training fold {fold+1}/{K}")
        model, scaler, val_loss = train_mdn_fold(X, y_log, occrecode_mapping, train_idx, val_idx, config, device)
        fold_models.append(model)
        scalers.append(scaler)
        val_losses.append(val_loss)
        torch.save(model.state_dict(), f'mdn_v2_fold_{fold}.pt')
    print("Val losses per fold:", val_losses, "mean:", np.mean(val_losses))

    # Prepare test data
    occ_col = 'occrecode'
    cont_cols = [c for c in X_test.columns if c != occ_col]
    Xc_test = X_test[cont_cols].values.astype(np.float32)
    occ_idx_test = X_test[occ_col].values.astype(np.int64)

    # MC sample from ensemble and back-transform
    N_MC = 2000  # larger MC for better approximation
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
        s = np.expm1(s)  # back to original scale
        all_samples = np.concatenate([all_samples, s], axis=1)

    # ensure exact N_MC samples
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
    print(f"Saved predictions.npy shape {all_samples.shape}")

    # OOB CRPS (back-transform y)
    print("Computing OOB CRPS...")
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
        s = np.expm1(s)
        oob_samples[val_idx] = s
        got[val_idx] = True

    assert got.all(), "OOB coverage incomplete!"
    y_orig = np.expm1(y_log.values)
    crps_scores = crps_ensemble(oob_samples, y_orig)
    mean_crps = crps_scores.mean()
    print(f"Mean OOB CRPS: {mean_crps:.5f}")
    return all_samples, mean_crps

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    X_trn = "X_trn.csv"
    Y_trn = "Y_trn.csv"
    X_tst = "X_test.csv"
    out_predictions = "predictions.npy"
    # choose number of optuna trials depending on time (e.g., 40 or 80 or 120)
    preds, mean_crps = run_full_pipeline(X_trn, Y_trn, X_tst, out_predictions=out_predictions, device_str='cuda', n_trials=40)
    print("Done. Mean OOB CRPS:", mean_crps)
