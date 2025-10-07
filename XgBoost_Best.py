# Block: imports
import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from scipy.interpolate import PchipInterpolator
from scipy.stats import rankdata, norm
from properscoring import crps_ensemble

# Block: fast settings
SEED = 42
USE_GPU = False
EARLY_STOP = 150
N_SAMPLES = 700
SEED_LIST = [42, 1337]

# Block: the 3 best configs you found (we’ll ensemble them)
CFG = [
    dict(max_depth=4, learning_rate=0.03, min_child_weight=5, subsample=0.70, colsample_bytree=0.70, gamma=0, reg_alpha=0.5, n_estimators=4000),
    dict(max_depth=4, learning_rate=0.03, min_child_weight=5, subsample=0.85, colsample_bytree=0.70, gamma=0, reg_alpha=0.5, n_estimators=4000),
    dict(max_depth=4, learning_rate=0.05, min_child_weight=5, subsample=0.85, colsample_bytree=0.70, gamma=0, reg_alpha=0.5, n_estimators=4000),
]

# Block: quantile levels (focus around center, with tails)
TAUS = np.r_[0.001, 0.005, np.linspace(0.01, 0.99, 39), 0.995, 0.999]

# Block: load data and split
DATA_DIR = Path(r"C:\Users\Siem.Poppe\OneDrive - Rebelgroup\Overig\Quantitative Finance\Machine learning")
X = pd.read_csv(DATA_DIR / "X_trn.csv")
y = pd.read_csv(DATA_DIR / "y_trn.csv")["realrinc"]

cat_cols = ["occrecode", "wrkstat", "gender", "educcat", "maritalcat"]
num_cols = ["year", "age", "prestg10", "childs"]

pre = ColumnTransformer([
    ("num", "passthrough", num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=X["year"]
)

# Block: target transform (we train on log1p, predict back with expm1)
def t_y(a):
    return np.log1p(a)

def inv_t(a):
    return np.expm1(a)

# Block: make xgboost params for one tau
def make_params(tau, cfg, seed):
    return dict(
        objective="reg:quantileerror",
        quantile_alpha=float(tau),
        max_depth=cfg["max_depth"],
        eta=cfg["learning_rate"],
        min_child_weight=cfg["min_child_weight"],
        subsample=cfg["subsample"],
        colsample_bytree=cfg["colsample_bytree"],
        gamma=cfg["gamma"],
        alpha=cfg["reg_alpha"],
        reg_lambda=1.0,
        tree_method="gpu_hist" if USE_GPU else "hist",
        random_state=int(seed),
    )

# Block: train a set of tau-models once and return their validation quantiles on original scale
def fit_quantiles_once(X_tr, y_tr, X_va, y_va, taus, cfg, seed):
    Xtr_t = pre.fit_transform(X_tr)
    Xva_t = pre.transform(X_va)
    ytr = t_y(y_tr)
    yva = t_y(y_va)
    dtr = xgb.DMatrix(Xtr_t, label=ytr)
    dva = xgb.DMatrix(Xva_t, label=yva)
    Q = {}
    for tau in taus:
        params = make_params(tau, cfg, seed)
        bst = xgb.train(
            params=params,
            dtrain=dtr,
            num_boost_round=cfg["n_estimators"],
            evals=[(dva, "valid")],
            early_stopping_rounds=EARLY_STOP,
            verbose_eval=False,
        )
        best_iter = bst.best_iteration if bst.best_iteration is not None else bst.best_ntree_limit - 1
        pred_t = bst.predict(dva, iteration_range=(0, best_iter + 1))
        Q[float(tau)] = inv_t(pred_t)
    return Q

# Block: force quantiles to be non-decreasing across taus
def enforce_monotone(Q):
    ts = sorted(Q.keys())
    M = np.vstack([Q[t] for t in ts])
    M = np.maximum.accumulate(M, axis=0)
    return {t: M[i] for i, t in enumerate(ts)}

# Block: sample from the quantile curves with PCHIP interpolation (optionally warped by isotonic)
def sample_pchip(Q, taus, n_draws=N_SAMPLES, seed=SEED, tau_inverse=None):
    ts = np.array(sorted(Q.keys()))
    Qmat = np.vstack([Q[t] for t in ts]).T
    rng = np.random.default_rng(seed)
    U = rng.uniform(0, 1, size=(Qmat.shape[0], n_draws))
    if tau_inverse is not None:
        U = tau_inverse(U)
    out = np.empty_like(U)
    for i in range(Qmat.shape[0]):
        p = PchipInterpolator(ts, Qmat[i], extrapolate=True)
        out[i] = p(U[i])
    return np.clip(out, 0.0, None)

# Block: fit isotonic calibration and return a strictly-increasing inverse CDF warp
def strict_isotonic_inverse(y_true, Q, taus, eps=1e-3):
    ts = np.array(sorted(taus))
    M = np.vstack([Q[t] for t in ts]).T
    yv = np.asarray(y_true, float)
    idx = (M <= yv[:, None]).sum(axis=1) - 1
    idx = np.clip(idx, 0, len(ts) - 2)
    row = np.arange(M.shape[0])
    ql = M[row, idx]
    qh = M[row, idx + 1]
    tl = ts[idx]
    th = ts[idx + 1]
    w = np.clip((yv - ql) / np.maximum(qh - ql, 1e-12), 0.0, 1.0)
    pit = tl + w * (th - tl)
    ranks = rankdata(pit, method="average") / (len(pit) + 1.0)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True)
    iso.fit(pit, ranks)
    grid = np.linspace(0, 1, 4001)
    G = iso.predict(grid)
    G = (1 - eps) * G + eps * grid
    G = np.clip(G, 0.0, 1.0)
    G = np.maximum.accumulate(G)
    Gu, idxu = np.unique(G, return_index=True)
    xu = grid[idxu]
    if len(Gu) < 3 or not np.all(np.isfinite(Gu)) or not np.all(np.isfinite(xu)):
        return lambda U: np.asarray(U, float)
    inv = PchipInterpolator(Gu, xu, extrapolate=True)
    return lambda U: np.clip(inv(np.clip(np.asarray(U, float), 0.0, 1.0)), 0.0, 1.0)

# Block: coverage and CRPS helpers
def coverage(y_true, samples, levels=(0.5, 0.8, 0.9)):
    cov = {}
    for q in levels:
        lo = np.quantile(samples, (1 - q) / 2, axis=1)
        hi = np.quantile(samples, 1 - (1 - q) / 2, axis=1)
        cov[q] = float(np.mean((y_true >= lo) & (y_true <= hi)))
    return cov

def mean_crps(y_true, samples):
    row = crps_ensemble(np.asarray(y_true), np.asarray(samples))
    return float(np.mean(row)), row

# Block: fit the ensemble across your 3 configs × 2 seeds and average their quantiles
def fit_ensemble_Q(X_tr, y_tr, X_va, y_va, taus):
    Q_sum = None
    for cfg in CFG:
        for s in SEED_LIST:
            Q = fit_quantiles_once(X_tr, y_tr, X_va, y_va, taus, cfg, s)
            Q = enforce_monotone(Q)
            if Q_sum is None:
                Q_sum = {t: Q[t].copy() for t in Q}
            else:
                for t in Q:
                    Q_sum[t] += Q[t]
    for t in Q_sum:
        Q_sum[t] /= (len(CFG) * len(SEED_LIST))
    return enforce_monotone(Q_sum)

# Block: simple width scalers (symmetric and asymmetric around the median)
def scale_sym(Q, s):
    ts = sorted(Q.keys())
    t0 = min(ts, key=lambda t: abs(t - 0.5))
    med = Q[t0]
    out = {}
    for t in ts:
        out[t] = med + s * (Q[t] - med)
    return enforce_monotone(out)

def scale_asym(Q, s_lo, s_hi):
    ts = sorted(Q.keys())
    t0 = min(ts, key=lambda t: abs(t - 0.5))
    med = Q[t0]
    out = {}
    for t in ts:
        if t < t0:
            out[t] = med + s_lo * (Q[t] - med)
        elif t > t0:
            out[t] = med + s_hi * (Q[t] - med)
        else:
            out[t] = med.copy()
    return enforce_monotone(out)

# Block: small width search (first symmetric, then asymmetric)
def tune_widths(y_true, Q, taus, tau_inv, n_draws=N_SAMPLES, seed=SEED):
    sym_grid = [0.98, 1.00, 1.02, 1.04, 1.06]
    best = (1.0, 9e18, "sym")
    for s in sym_grid:
        Qs = scale_sym(Q, s)
        smp = sample_pchip(Qs, taus, n_draws=n_draws, seed=seed, tau_inverse=tau_inv)
        crps, _ = mean_crps(y_true, smp)
        if crps < best[1]:
            best = (s, crps, "sym")
    s_star = best[0]
    lo_grid = [0.95, 1.00, 1.05]
    hi_grid = [1.00, 1.05, 1.10]
    best_asym = (1.0, 1.0, best[1], "asym")
    for slo in lo_grid:
        for shi in hi_grid:
            Qg = scale_asym(Q, slo, shi)
            smp = sample_pchip(Qg, taus, n_draws=n_draws, seed=seed, tau_inverse=tau_inv)
            crps, _ = mean_crps(y_true, smp)
            if crps < best_asym[2]:
                best_asym = (slo, shi, crps, "asym")
    if best_asym[2] < best[1]:
        return ("asym", dict(s_lo=best_asym[0], s_hi=best_asym[1], crps=best_asym[2]))
    else:
        return ("sym", dict(s=s_star, crps=best[1]))

# Block: light per-group width tuning on wrkstat
def tune_group_wrkstat(X_va, Q, fallback, min_n=600):
    if "wrkstat" not in X_va.columns:
        n = len(X_va)
        return np.full(n, fallback["s_lo"], float), np.full(n, fallback["s_hi"], float), np.zeros(n)
    wrk = X_va["wrkstat"].astype("category")
    counts = wrk.value_counts()
    big_groups = counts[counts >= min_n].index.tolist()
    n = len(wrk)
    slo_row = np.full(n, fallback["s_lo"], float)
    shi_row = np.full(n, fallback["s_hi"], float)
    b_row = np.zeros(n, float)
    lo_grid = [0.95, 1.00, 1.05]
    hi_grid = [1.00, 1.05, 1.12]
    ts = sorted(Q.keys())
    t0 = min(ts, key=lambda t: abs(t - 0.5))
    for g in big_groups:
        mask = (wrk.values == g)
        y_sub = y_valid.to_numpy()[mask]
        if y_sub.size < min_n:
            continue
        Q_sub = {t: Q[t][mask] for t in Q}
        best = (fallback["s_lo"], fallback["s_hi"], 9e18)
        for slo in lo_grid:
            for shi in hi_grid:
                Qg = scale_asym(Q_sub, slo, shi)
                smp = sample_pchip(Qg, TAUS, n_draws=400, seed=SEED, tau_inverse=None)
                crps, _ = mean_crps(y_sub, smp)
                if crps < best[2]:
                    best = (slo, shi, crps)
        slo_row[mask] = best[0]
        shi_row[mask] = best[1]
        b_row[mask] = 0.0
    return slo_row, shi_row, b_row

# Block: turn quantiles into a rough LogNormal (no training), then sample
def quantiles_to_lognormal(Q, taus):
    ts = sorted(Q.keys())
    t50 = min(ts, key=lambda t: abs(t - 0.5))
    t16 = min(ts, key=lambda t: abs(t - 0.16))
    t84 = min(ts, key=lambda t: abs(t - 0.84))
    q50 = np.clip(Q[t50], 1e-12, None)
    q16 = np.clip(Q[t16], 1e-12, None)
    q84 = np.clip(Q[t84], 1e-12, None)
    mu = np.log1p(q50)
    z84, z16 = norm.ppf(0.84), norm.ppf(0.16)
    denom = max(z84 - z16, 1e-6)
    s84 = np.log1p(q84)
    s16 = np.log1p(q16)
    sigma = np.clip((s84 - s16) / denom, 1e-3, 5.0)
    return mu, sigma

def sample_lognormal(mu, sigma, n_draws, seed):
    rng = np.random.default_rng(seed)
    Z = rng.normal(0.0, 1.0, size=(len(mu), n_draws))
    S = mu[:, None] + sigma[:, None] * Z
    return np.expm1(S).clip(0.0)

# Block: fit quantile ensemble
print("[S2-Quantile] Fit ensemble (your 3 configs × 2 seeds)…")
Q_ens = fit_ensemble_Q(X_train, y_train, X_valid, y_valid, TAUS)

# Block: isotonic calibration on the ensemble + baseline score
tau_inv = strict_isotonic_inverse(y_valid, Q_ens, TAUS, eps=1e-3)
smp0 = sample_pchip(Q_ens, TAUS, n_draws=N_SAMPLES, seed=SEED, tau_inverse=tau_inv)
crps0, _ = mean_crps(y_valid, smp0)
cov0 = coverage(y_valid.to_numpy(), smp0)
print("\n=== Quantile ensemble (strict-calibrated) ===")
print(f"Validation CRPS: {crps0:.2f}")
print(f"Coverage 50/80/90: {cov0[0.5]:.3f}/{cov0[0.8]:.3f}/{cov0[0.9]:.3f}")
print("Samples shape:", smp0.shape)

# Block: global width tuning and score
kind, info = tune_widths(y_valid.to_numpy(), Q_ens, TAUS, tau_inv, n_draws=N_SAMPLES, seed=SEED)
if kind == "sym":
    Q_tuned = scale_sym(Q_ens, info["s"])
    tag = f"Sym s*={info['s']:.3f}"
else:
    Q_tuned = scale_asym(Q_ens, info["s_lo"], info["s_hi"])
    tag = f"Asym (s_lo,s_hi)=({info['s_lo']:.3f},{info['s_hi']:.3f})"
smp_tuned = sample_pchip(Q_tuned, TAUS, n_draws=N_SAMPLES, seed=SEED, tau_inverse=tau_inv)
crps_tuned, _ = mean_crps(y_valid, smp_tuned)
cov_tuned = coverage(y_valid.to_numpy(), smp_tuned)
print("\n=== Quantile + global width (strict-calibrated) ===")
print(f"{tag} | CRPS: {crps_tuned:.2f}")
print(f"Coverage 50/80/90: {cov_tuned[0.5]:.3f}/{cov_tuned[0.8]:.3f}/{cov_tuned[0.9]:.3f}")

# Block: per-row widths from wrkstat and score
fallback = dict(s_lo=1.0, s_hi=1.0)
if kind == "asym":
    fallback = dict(s_lo=info["s_lo"], s_hi=info["s_hi"])
slo_row, shi_row, b_row = tune_group_wrkstat(X.loc[X_valid.index], Q_tuned, fallback, min_n=600)
ts = sorted(Q_tuned.keys())
t0 = min(ts, key=lambda t: abs(t - 0.5))
med = Q_tuned[t0]
Q_row = {}
for t in ts:
    delta = Q_tuned[t] - med
    Q_row[t] = med + (slo_row * (t < t0) + shi_row * (t > t0)) * delta
Q_row = enforce_monotone(Q_row)
smp_row = sample_pchip(Q_row, TAUS, n_draws=N_SAMPLES, seed=SEED, tau_inverse=tau_inv)
crps_row, _ = mean_crps(y_valid, smp_row)
cov_row = coverage(y_valid.to_numpy(), smp_row)
print("\n=== Quantile + per-group wrkstat (strict-calibrated) ===")
print(f"CRPS: {crps_row:.2f}")
print(f"Coverage 50/80/90: {cov_row[0.5]:.3f}/{cov_row[0.8]:.3f}/{cov_row[0.9]:.3f}")

# Block: derived LogNormal small blend (tries a few tiny tweaks and mixes)
mu0, sg0 = quantiles_to_lognormal(Q_row, TAUS)
temps = [0.95, 1.00, 1.05]
mbias = [0.0, 0.01, -0.01]
mix_w = [0.0, 0.2, 0.4]
best_mix = (crps_row, 0.0, 1.00, 0.0)
for t in temps:
    for b in mbias:
        mu = mu0 * (1.0 + b)
        sg = np.clip(sg0 * t, 1e-3, 5.0)
        smp_ln = sample_lognormal(mu, sg, N_SAMPLES, SEED)
        for w in mix_w:
            smp_mix = (1.0 - w) * smp_row + w * smp_ln
            crps_m, _ = mean_crps(y_valid, smp_mix)
            if crps_m < best_mix[0]:
                best_mix = (crps_m, w, t, b)
crps_final = best_mix[0]
w_star, t_star, b_star = best_mix[1], best_mix[2], best_mix[3]
mu = mu0 * (1.0 + b_star)
sg = np.clip(sg0 * t_star, 1e-3, 5.0)
smp_ln = sample_lognormal(mu, sg, N_SAMPLES, SEED)
smp_final = (1.0 - w_star) * smp_row + w_star * smp_ln
cov_final = coverage(y_valid.to_numpy(), smp_final)
print("\n=== Final (Quantile tuned + small LogNormal blend) ===")
print(f"CRPS: {crps_final:.2f} | mix w={w_star:.2f}, sigma-temp={t_star:.2f}, mu-bias={b_star:+.3f}")
print(f"Coverage 50/80/90: {cov_final[0.5]:.3f}/{cov_final[0.8]:.3f}/{cov_final[0.9]:.3f}")
print("Samples shape:", smp_final.shape)
