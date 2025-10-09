# -*- coding: utf-8 -*-

import os, json, logging
from time import perf_counter
from contextlib import contextmanager
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from scipy.interpolate import PchipInterpolator
from scipy.stats import rankdata, kstest
from properscoring import crps_ensemble

import xgboost as xgb
from ngboost import NGBRegressor
from ngboost.distns import Normal, LogNormal
from ngboost.scores import LogScore
from sklearn.tree import DecisionTreeRegressor

# ----------------------------- Paths & logging -----------------------------
ART_DIR = Path("artifacts"); ART_DIR.mkdir(exist_ok=True)
LOG_DIR = Path("logs"); LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(timestamp)s | %(levelname)s | %(message)s".replace("timestamp", "asctime"),
    handlers=[logging.FileHandler(LOG_DIR / "crps_pushdown_v3_tuned.log", mode="w", encoding="utf-8"),
              logging.StreamHandler()],
)
log = logging.getLogger("crps")
_TIMES = {}
@contextmanager
def timed(name):
    t0 = perf_counter(); log.info(f"[START] {name}")
    yield
    dt = perf_counter() - t0; _TIMES[name] = dt; log.info(f"[DONE ] {name} in {dt:0.2f}s")
def dump_stage_times():
    if not _TIMES: return
    total = sum(_TIMES.values())
    log.info("=== Stage timings ===")
    for k, v in _TIMES.items(): log.info(f"{k:<40} {v:7.2f}s")
    log.info(f"{'-'*40} --------"); log.info(f"{'TOTAL':<40} {total:7.2f}s")

# ----------------------------- Toggles & Profiles -----------------------------
PROFILE = "FAST"   # FAST / MEDIUM / FULL
DO_XGB_TUNE = True                 # <--- turn tuning ON/OFF

# Coarse XGB grid (tight, high-leverage) + tune settings
XGB_GRID = dict(
    max_depth=[3, 4, 5],
    eta=[0.025, 0.03, 0.05],
    subsample=[0.75, 0.85],
    colsample_bytree=[0.60, 0.75],
    min_child_weight=[2, 5, 8],
)
XGB_N_ESTIM_TUNE = 2600
SEEDS_TUNE       = [42, 1337]     # seed bagging during tuning
N_DRAWS_TUNE     = 220
TAUS_TUNE        = np.r_[0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                         0.60, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99, 0.995]
TOPK_REFINE      = 3              # refine the best K on full TAUS

if PROFILE == "FAST":
    SEED = 42
    EARLY_STOP = 120
    TAUS = np.r_[0.01, 0.025, 0.05, 0.10,
                 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.85, 0.90, 0.92, 0.94, 0.96,
                 0.975, 0.985, 0.99, 0.995, 0.997]  # 21 knots tilted to top end
    N_DRAWS_BASE  = 350
    N_DRAWS_FINAL = 2200
    # fallback XGB if tuning off
    XGB_FALLBACK = dict(max_depth=4, eta=0.03, min_child_weight=5,
                        subsample=0.85, colsample_bytree=0.70,
                        gamma=0, alpha=0.5, n_estimators=3000)
elif PROFILE == "MEDIUM":
    SEED = 42; EARLY_STOP = 150
    TAUS = np.r_[0.005, 0.01, np.linspace(0.03, 0.97, 25), 0.99, 0.995, 0.997]
    N_DRAWS_BASE = 600; N_DRAWS_FINAL = 2400
    XGB_FALLBACK = dict(max_depth=4, eta=0.03, min_child_weight=5,
                        subsample=0.85, colsample_bytree=0.70,
                        gamma=0, alpha=0.5, n_estimators=3500)
else:
    SEED = 42; EARLY_STOP = 180
    TAUS = np.r_[0.001, 0.005, np.linspace(0.01, 0.99, 39), 0.995, 0.997, 0.999]
    N_DRAWS_BASE = 800; N_DRAWS_FINAL = 2600
    XGB_FALLBACK = dict(max_depth=4, eta=0.03, min_child_weight=5,
                        subsample=0.70, colsample_bytree=0.70,
                        gamma=0, alpha=0.5, n_estimators=4000)

USE_GPU_XGB = False   # set True if GPU available and xgboost supports it

# ----------------------------- Data -----------------------------
DATA_DIR = Path(r"C:\Users\Siem.Poppe\OneDrive - Rebelgroup\Overig\Quantitative Finance\Machine learning")
X = pd.read_csv(DATA_DIR / "X_trn.csv")
y = pd.read_csv(DATA_DIR / "y_trn.csv")["realrinc"]
categorical_cols = ["occrecode", "wrkstat", "gender", "educcat", "maritalcat"]
numeric_cols     = ["year", "age", "prestg10", "childs"]
pre = ColumnTransformer([("num", "passthrough", numeric_cols),
                         ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)])
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=X["year"])

# ----------------------------- Helpers -----------------------------
def _rng(seed=SEED): return np.random.default_rng(int(seed))
def t_y(y):   return np.log1p(y)
def inv_t(z): return np.expm1(z)

def coverage(y_true, samples, levels=(0.5, 0.8, 0.9)):
    cov = {}; yv = np.asarray(y_true, float); S = np.asarray(samples)
    for q in levels:
        lo = np.quantile(S, (1-q)/2, axis=1); hi = np.quantile(S, 1-(1-q)/2, axis=1)
        cov[q] = float(np.mean((yv >= lo) & (yv <= hi)))
    return cov

def crps_mean(y_true, samples):
    row = crps_ensemble(np.asarray(y_true), np.asarray(samples))
    return float(np.mean(row)), row

def enforce_monotone(Q_dict):
    ts = sorted(Q_dict.keys()); M = np.vstack([Q_dict[t] for t in ts])
    M = np.maximum.accumulate(M, axis=0)
    return {t: M[i] for i, t in enumerate(ts)}

def strict_isotonic_inverse_from_quantiles(y_true, Q_dict, taus, eps=1e-3):
    ts = np.array(sorted(taus), float)
    Q  = np.vstack([Q_dict[t] for t in ts]).T
    yv = np.asarray(y_true, float)
    idx = (Q <= yv[:, None]).sum(axis=1) - 1
    idx = np.clip(idx, 0, len(ts) - 2)
    row = np.arange(Q.shape[0])
    ql, qh = Q[row, idx], Q[row, idx + 1]
    tl, th = ts[idx], ts[idx + 1]
    w = np.clip((yv - ql) / np.maximum(qh - ql, 1e-12), 0.0, 1.0)
    pit = tl + w * (th - tl)
    emp = rankdata(pit, method="average") / (len(pit) + 1.0)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True); iso.fit(pit, emp)
    grid = np.linspace(0, 1, 4001); G = iso.predict(grid)
    G = (1 - eps) * G + eps * grid; G = np.maximum.accumulate(np.clip(G, 0.0, 1.0))
    Gu, idxu = np.unique(G, return_index=True); xu = grid[idxu]
    if len(Gu) < 3 or not np.all(np.isfinite(Gu)) or not np.all(np.isfinite(xu)):
        return lambda U: np.asarray(U, float)
    inv = PchipInterpolator(Gu, xu, extrapolate=True)
    return lambda U: np.clip(inv(np.clip(np.asarray(U, float), 0.0, 1.0)), 0.0, 1.0)

def sample_from_quantiles_pchip(Q_dict, taus, n_draws, seed, tau_inverse=None):
    ts = np.array(sorted(Q_dict.keys())); Qmat = np.vstack([Q_dict[t] for t in ts]).T
    rng = _rng(seed); U = rng.uniform(0, 1, size=(Qmat.shape[0], n_draws))
    if tau_inverse is not None: U = tau_inverse(U)
    out = np.empty_like(U)
    for i in range(Qmat.shape[0]):
        p = PchipInterpolator(ts, Qmat[i], extrapolate=True)
        out[i] = p(U[i])
    return np.clip(out, 0.0, None)

def scale_symmetric(Q, s):
    ts = sorted(Q.keys()); t0 = min(ts, key=lambda t: abs(t-0.5)); med = Q[t0]
    out = {t: med + s * (Q[t] - med) for t in ts}
    return enforce_monotone(out)

def scale_asymmetric(Q, s_lo, s_hi):
    ts = sorted(Q.keys()); t0 = min(ts, key=lambda t: abs(t-0.5)); med = Q[t0]; out = {}
    for t in ts:
        if t < t0:   out[t] = med + s_lo * (Q[t] - med)
        elif t > t0: out[t] = med + s_hi * (Q[t] - med)
        else:        out[t] = med.copy()
    return enforce_monotone(out)

def tiny_width_tune(y_true, Q, taus, tau_inv, n_draws, seed):
    ts = sorted(Q.keys()); t0 = min(ts, key=lambda t: abs(t-0.5)); med = Q[t0]
    def _apply(Qin, b_rel):
        shift = b_rel * med
        out = {t: np.clip(Qin[t] + shift, 0.0, None) for t in ts}
        return enforce_monotone(out)
    best_Q, best_c, best_tag = None, 1e18, "none"
    for s in [0.98, 1.00, 1.02, 1.04]:
        Qs = scale_symmetric(Q, s)
        for b in [-0.01, 0.0, 0.01]:
            Qsb = _apply(Qs, b)
            S = sample_from_quantiles_pchip(Qsb, taus, n_draws, seed, tau_inv)
            c, _ = crps_mean(y_true, S)
            if c < best_c: best_Q, best_c, best_tag = Qsb, c, f"sym s={s:.3f} b={b:+.3f}"
    for slo in [0.96, 1.00, 1.04]:
        for shi in [1.00, 1.04, 1.08]:
            Qa = scale_asymmetric(Q, slo, shi)
            for b in [-0.01, 0.0, 0.01]:
                Qab = _apply(Qa, b)
                S = sample_from_quantiles_pchip(Qab, taus, n_draws, seed, tau_inv)
                c, _ = crps_mean(y_true, S)
                if c < best_c: best_Q, best_c, best_tag = Qab, c, f"asym slo={slo:.3f} shi={shi:.3f} b={b:+.3f}"
    return best_Q, best_c, best_tag

def pinball_asym(y_true, Q_dict, taus, k_over=1.25, k_under=1.0):
    ts = sorted(Q_dict.keys()); yv = np.asarray(y_true, float); rows = []
    for tau in taus:
        t = float(min(ts, key=lambda t: abs(t - float(tau))))
        q = np.asarray(Q_dict[t], float)
        e = yv - q
        under = np.maximum(e, 0.0); over  = np.maximum(-e, 0.0)
        loss = k_under * t * under + k_over * (1.0 - t) * over
        rows.append(dict(tau=float(t), pinball=float(np.mean(loss)),
                         mean_under=float(np.mean(under)), mean_over=float(np.mean(over))))
    return pd.DataFrame(rows)

# ----------------------------- XGB & NGB -----------------------------
def _train_xgb_quantiles_singlecfg(X_tr, y_tr, X_va, y_va, taus, cfg, seeds, early_stop, use_gpu):
    """Train XGB quantiles for a single cfg across multiple seeds; average and enforce monotonicity."""
    Xtr_t = pre.fit_transform(X_tr); Xva_t = pre.transform(X_va)
    ytr = t_y(y_tr); yva = t_y(y_va)
    dtr = xgb.DMatrix(Xtr_t, label=ytr); dva = xgb.DMatrix(Xva_t, label=yva)
    Q_accum = None; total = 0
    for s in seeds:
        params = dict(
            objective="reg:quantileerror", quantile_alpha=0.5,
            max_depth=cfg["max_depth"], eta=cfg["eta"], min_child_weight=cfg["min_child_weight"],
            subsample=cfg["subsample"], colsample_bytree=cfg["colsample_bytree"],
            gamma=cfg.get("gamma", 0), alpha=cfg.get("alpha", 0.5), reg_lambda=1.0,
            tree_method="gpu_hist" if use_gpu else "hist", random_state=int(s),
        )
        Q_tau = {}
        for tau in taus:
            params["quantile_alpha"] = float(tau)
            bst = xgb.train(params=params, dtrain=dtr, num_boost_round=cfg["n_estimators"],
                            evals=[(dva, "valid")], early_stopping_rounds=early_stop, verbose_eval=False)
            best_iter = bst.best_iteration if bst.best_iteration is not None else bst.best_ntree_limit - 1
            pred_t = bst.predict(dva, iteration_range=(0, best_iter + 1))
            Q_tau[float(tau)] = inv_t(pred_t)
        Q_tau = enforce_monotone(Q_tau)
        if Q_accum is None: Q_accum = {t: Q_tau[t].copy() for t in Q_tau}
        else:
            for t in Q_tau: Q_accum[t] += Q_tau[t]
        total += 1
    for t in Q_accum: Q_accum[t] /= total
    return enforce_monotone(Q_accum)

def ngb_samples(X_tr, y_tr, X_va, mode, Dist, n_draws, seed, early_stop):
    Xtr_t = pre.fit_transform(X_tr); Xva_t = pre.transform(X_va)
    Base = DecisionTreeRegressor(max_depth=4, random_state=seed)
    ngb = NGBRegressor(Base=Base, Dist=Dist, Score=LogScore, natural_gradient=True,
                       learning_rate=0.05, n_estimators=1600, minibatch_frac=0.5,
                       random_state=seed, verbose=False, early_stopping_rounds=early_stop, validation_fraction=None)
    if mode == "raw":
        ytr = np.clip(np.asarray(y_tr, float), 1e-8, None)
        yv  = np.clip(np.asarray(y_va, float), 1e-8, None)
        ngb.fit(Xtr_t, ytr, X_val=Xva_t, Y_val=yv); dist = ngb.pred_dist(Xva_t)
        # sampling fallback logic
        if hasattr(dist, "ppf"):
            rng = _rng(seed); U = rng.uniform(0.0, 1.0, size=(len(y_va), n_draws))
            try: return np.clip(dist.ppf(U), 0.0, None)
            except Exception: pass
        loc, scale = np.asarray(dist.params)[:,0], np.asarray(dist.params)[:,1]
        rng = _rng(seed); Z = rng.normal(0.0, 1.0, size=(len(loc), n_draws))
        return np.exp(loc[:,None] + scale[:,None]*Z).clip(0.0)
    else:
        ztr = t_y(y_tr); zv = t_y(y_va)
        ngb.fit(Xtr_t, ztr, X_val=Xva_t, Y_val=zv); dist = ngb.pred_dist(Xva_t)
        if hasattr(dist, "ppf"):
            rng = _rng(seed); U = rng.uniform(0.0, 1.0, size=(len(y_va), n_draws))
            try: return np.expm1(dist.ppf(U)).clip(0.0)
            except Exception: pass
        loc, scale = np.asarray(dist.params)[:,0], np.asarray(dist.params)[:,1]
        rng = _rng(seed); Z = rng.normal(0.0, 1.0, size=(len(loc), n_draws))
        return np.expm1(loc[:,None] + scale[:,None]*Z).clip(0.0)

# ----------------------------- Per-group wrkstat -----------------------------
def pergroup_asym_wrkstat(X_va_full, y_va, Q_base, taus, tau_inv, seed=42, min_n=600):
    Xva = X_va_full.copy()
    if "wrkstat" not in Xva.columns:
        n = len(y_va); return np.ones(n), np.ones(n)
    vals = Xva["wrkstat"].astype("category"); counts = vals.value_counts()
    big = counts[counts >= min_n].index.tolist()
    slo_row = np.ones(len(vals), float); shi_row = np.ones(len(vals), float)
    lo_grid = [0.96, 1.00, 1.04]; hi_grid = [1.00, 1.04, 1.08]
    ts = sorted(Q_base.keys())
    for g in big:
        mask = (vals.values == g); y_sub = y_va.to_numpy()[mask]
        Q_sub = {t: Q_base[t][mask] for t in ts}
        best = (1.0, 1.0, 1e18)
        for slo in lo_grid:
            for shi in hi_grid:
                Qa = scale_asymmetric(Q_sub, slo, shi)
                S = sample_from_quantiles_pchip(Qa, taus, n_draws=300, seed=seed, tau_inverse=tau_inv)
                c, _ = crps_mean(y_sub, S)
                if c < best[2]: best = (slo, shi, c)
        slo_row[mask], shi_row[mask] = best[0], best[1]
    return slo_row, shi_row

# ----------------------------- Decile-conditional width/bias -----------------------------
def decile_conditional_tune(y_true, Q, taus, tau_inv, n_draws, seed, n_bins=10):
    ts = sorted(Q.keys()); t0 = min(ts, key=lambda t: abs(t - 0.5)); med = Q[t0]
    edges = np.quantile(med, np.linspace(0, 1, n_bins + 1)); which = np.digitize(med, edges[1:-1], right=True)
    lo_grid = [0.96, 1.00, 1.04]; hi_grid = [1.00, 1.06, 1.12, 1.16]; b_grid = [-0.01, 0.00, 0.01]
    slo_row = np.ones(len(med), float); shi_row = np.ones(len(med), float); b_row = np.zeros(len(med), float)
    for b_idx in range(n_bins):
        mask = which == b_idx
        if mask.sum() < 120: continue
        y_sub = np.asarray(y_true, float)[mask]
        Q_sub = {t: Q[t][mask] for t in ts}
        best = (1.0, 1.0, 0.0, 1e18)
        med_sub = Q_sub[t0]
        for slo in lo_grid:
            for shi in hi_grid:
                for b in b_grid:
                    shift = b * med_sub; Q_try = {}
                    for t in ts:
                        if t < 0.5: scale = slo
                        elif t > 0.5: scale = shi
                        else: scale = 1.0
                        Q_try[t] = np.clip(med_sub + scale * (Q_sub[t] - med_sub) + shift, 0.0, None)
                    Q_try = enforce_monotone(Q_try)
                    S_sub = sample_from_quantiles_pchip(Q_try, taus, n_draws=250, seed=seed, tau_inverse=tau_inv)
                    c, _ = crps_mean(y_sub, S_sub)
                    if c < best[3]: best = (slo, shi, b, c)
        slo_row[mask], shi_row[mask], b_row[mask] = best[0], best[1], best[2]
    Q_out = {}
    for t in ts:
        if t < 0.5: scale = slo_row
        elif t > 0.5: scale = shi_row
        else: scale = np.ones_like(slo_row)
        Q_out[t] = np.clip(med + scale * (Q[t] - med) + b_row * med, 0.0, None)
    return enforce_monotone(Q_out)

# ----------------------------- GPD tail splice (with autotune) -----------------------------
def _solve_xi_beta_two_points(u, q1, p1, q2, p2, p0, max_iter=10):
    r1 = (p1 - p0) / (1.0 - p0)
    r2 = (p2 - p0) / (1.0 - p0)
    y1 = np.maximum(q1 - u, 1e-9)
    y2 = np.maximum(q2 - u, 1e-9)
    xi = np.full_like(u, 0.2, dtype=float)
    for _ in range(max_iter):
        A1 = (1.0 - r1) ** (-xi) - 1.0
        A2 = (1.0 - r2) ** (-xi) - 1.0
        ratio = np.clip(y2 / np.maximum(y1, 1e-12), 1.0 + 1e-9, 1e9)
        f = (A2 / np.maximum(A1, 1e-12)) - ratio
        dA1 = -np.log(1.0 - r1) * (1.0 - r1) ** (-xi)
        dA2 = -np.log(1.0 - r2) * (1.0 - r2) ** (-xi)
        df = (dA2 * A1 - A2 * dA1) / (np.maximum(A1, 1e-12) ** 2)
        step = np.clip(-f / np.maximum(np.abs(df), 1e-6), -0.5, 0.5)
        xi = np.clip(xi + step, -0.3, 1.8)
    A1 = (1.0 - r1) ** (-xi) - 1.0
    beta = xi * y1 / np.maximum(A1, 1e-9)
    beta = np.clip(beta, 1e-6, 1e12)
    bad = ~np.isfinite(xi) | ~np.isfinite(beta) | (beta <= 0)
    xi = np.where(bad, 0.1, xi)
    beta = np.where(bad, np.maximum(y1, 1e-6), beta)
    return xi, beta

def _gpd_quantile(u, xi, beta, r):
    r = np.clip(r, 1e-9, 1 - 1e-9)
    if np.isscalar(xi) or np.ndim(xi) == 0:
        return u + (beta * np.log(1.0 / (1.0 - r)) if abs(xi) < 1e-6 else (beta / xi) * (np.power(1.0 - r, -xi) - 1.0))
    out = np.empty((len(u),) + np.shape(r)[1:], dtype=float)
    small = np.abs(xi) < 1e-6; big = ~small
    if np.any(small): out[small] = u[small, None] + beta[small, None] * np.log(1.0 / (1.0 - r))
    if np.any(big):   out[big]   = u[big, None] + (beta[big, None] / xi[big, None]) * (np.power(1.0 - r, -xi[big, None]) - 1.0)
    return out

def sample_with_gpd_splice(Q_dict, taus, n_draws, seed, tau_inverse=None, p0=0.95, p1=0.99, p2=0.997, gamma=1.0):
    ts = np.array(sorted(Q_dict.keys())); Qmat = np.vstack([Q_dict[t] for t in ts]).T
    n = Qmat.shape[0]
    pchips = [PchipInterpolator(ts, Qmat[i], extrapolate=True) for i in range(n)]
    q_p0 = np.array([p(p0) for p in pchips]); q_p1 = np.array([p(p1) for p in pchips]); q_p2 = np.array([p(p2) for p in pchips])
    xi, beta = _solve_xi_beta_two_points(q_p0, q_p1, p1, q_p2, p2, p0)
    rng = np.random.default_rng(int(seed))
    U = rng.uniform(0.0, 1.0, size=(n, n_draws))
    if tau_inverse is not None: U = tau_inverse(U)
    S = np.empty_like(U)
    for i in range(n):
        mask_body = U[i] <= p0
        if np.any(mask_body): S[i, mask_body] = pchips[i](U[i, mask_body])
        if np.any(~mask_body):
            r = (U[i, ~mask_body] - p0) / (1.0 - p0)
            if gamma != 1.0: r = np.power(r, gamma)
            S[i, ~mask_body] = _gpd_quantile(q_p0[i], xi[i], beta[i], r)
    return np.clip(S, 0.0, None)

def autotune_tail(Q_dict, y_true, taus, tau_inverse, seed, draws=280):
    grid_p0   = [0.94, 0.95, 0.96]
    grid_p2   = [0.997, 0.998]
    grid_gam  = [0.85, 0.90, 1.00]
    best = (None, 1e18, (None, None, None))
    for p0 in grid_p0:
        for p2 in grid_p2:
            for gam in grid_gam:
                S = sample_with_gpd_splice(Q_dict, taus, n_draws=draws, seed=seed, tau_inverse=tau_inverse,
                                           p0=p0, p1=0.99, p2=p2, gamma=gam)
                c, _ = crps_mean(y_true, S)
                if c < best[1]: best = (S, c, (p0, p2, gam))
    return best

# ----------------------------- Blending utils -----------------------------
def simplex_grid(n_models, step=0.25):
    vals = np.arange(0.0, 1.0 + 1e-9, step)
    if n_models == 1:
        yield np.array([1.0]); return
    def rec(prefix, remaining, k):
        if k == 1:
            v = np.array(prefix + [remaining])
            if abs(v.sum() - 1.0) < 1e-9: yield v
            return
        for a in vals:
            if a <= remaining + 1e-9:
                yield from rec(prefix + [a], round(remaining - a, 10), k-1)
    yield from rec([], 1.0, n_models)

def coordinate_descent(w0, eval_fn, iters=10, step=0.05):
    w = w0.copy(); base = eval_fn(w)
    for _ in range(iters):
        improved = False
        for i in range(len(w)):
            for delta in (+step, -step):
                w_try = w.copy(); w_try[i] = np.clip(w_try[i] + delta, 0.0, 1.0)
                s = w_try.sum()
                if s == 0: continue
                w_try /= s
                val = eval_fn(w_try)
                if val < base: w, base, improved = w_try, val, True
        if not improved: break
    return w, base

# ----------------------------- XGB tuning (coarse → refine) -----------------------------
def xgb_cfg_iter(grid):
    keys = list(grid.keys())
    for vals in product(*[grid[k] for k in keys]):
        cfg = dict(zip(keys, vals))
        cfg["gamma"] = 0; cfg["alpha"] = 0.5; cfg["n_estimators"] = XGB_N_ESTIM_TUNE
        yield cfg

def score_xgb_cfg(cfg, taus_for_tune, seeds, early_stop, use_gpu, n_draws=N_DRAWS_TUNE):
    Q = _train_xgb_quantiles_singlecfg(X_tr, y_tr, X_va, y_va, taus_for_tune, cfg, seeds, early_stop, use_gpu)
    tau_inv = strict_isotonic_inverse_from_quantiles(y_va, Q, taus_for_tune, eps=1e-3)
    # cheap tiny width/bias (helps proxy fidelity)
    Q_tiny, _, _ = tiny_width_tune(y_va, Q, taus_for_tune, tau_inv, n_draws=n_draws, seed=SEED)
    S = sample_from_quantiles_pchip(Q_tiny, taus_for_tune, n_draws=n_draws, seed=SEED, tau_inverse=tau_inv)
    c, _ = crps_mean(y_va, S)
    return c

# ----------------------------- Pipeline -----------------------------
def main():
    log.info(f"PROFILE={PROFILE} | GPU(XGB)={USE_GPU_XGB}")
    log.info(f"TAUS={len(TAUS)}, EARLY_STOP={EARLY_STOP}, N_DRAWS_BASE={N_DRAWS_BASE}, N_DRAWS_FINAL={N_DRAWS_FINAL}")
    # 0) XGB base: tune or fallback
    with timed("XGB coarse+refine hyperparameter tuning" if DO_XGB_TUNE else "XGB (fallback config)"):
        if DO_XGB_TUNE:
            results = []
            for cfg in xgb_cfg_iter(XGB_GRID):
                c = score_xgb_cfg(cfg, TAUS_TUNE, SEEDS_TUNE, EARLY_STOP, USE_GPU_XGB, n_draws=N_DRAWS_TUNE)
                results.append((c, cfg))
            results.sort(key=lambda x: x[0])
            log.info(f"XGB coarse top5 proxy CRPS: {[round(c,2) for c,_ in results[:5]]}")

            refined = []
            for c0, cfg0 in results[:TOPK_REFINE]:
                # promote to full TAUS; allow a bit more rounds if fallback was larger
                cfg_ref = cfg0.copy()
                cfg_ref["n_estimators"] = max(cfg_ref["n_estimators"], XGB_FALLBACK["n_estimators"])
                Q_try = _train_xgb_quantiles_singlecfg(X_tr, y_tr, X_va, y_va, TAUS, cfg_ref, SEEDS_TUNE, EARLY_STOP, USE_GPU_XGB)
                tau_inv_try = strict_isotonic_inverse_from_quantiles(y_va, Q_try, TAUS, eps=1e-3)
                S_try = sample_from_quantiles_pchip(Q_try, TAUS, n_draws=N_DRAWS_TUNE, seed=SEED, tau_inverse=tau_inv_try)
                c_try, _ = crps_mean(y_va, S_try)
                refined.append((c_try, cfg_ref, Q_try))
            refined.sort(key=lambda x: x[0])
            c_best, XGB_CFG_BEST, Q_xgb = refined[0]
            log.info(f"XGB tuned cfg: {XGB_CFG_BEST} (proxy CRPS={c_best:.2f})")
        else:
            XGB_CFG_BEST = XGB_FALLBACK
            Q_xgb = _train_xgb_quantiles_singlecfg(X_tr, y_tr, X_va, y_va, TAUS, XGB_CFG_BEST, [SEED], EARLY_STOP, USE_GPU_XGB)

    # 1) PIT calibrator
    with timed("PIT calibration (isotonic inverse)"):
        tau_inv = strict_isotonic_inverse_from_quantiles(y_va, Q_xgb, TAUS, eps=1e-3)

    # 2) Tiny width + bias (global)
    with timed("XGB sampling + width tuning"):
        S_xgb0 = sample_from_quantiles_pchip(Q_xgb, TAUS, n_draws=N_DRAWS_BASE, seed=SEED, tau_inverse=tau_inv)
        c0, _ = crps_mean(y_va, S_xgb0); cov0 = coverage(y_va, S_xgb0)
        log.info(f"XGB baseline: CRPS={c0:.2f} | Cov50/80/90={cov0[0.5]:.3f}/{cov0[0.8]:.3f}/{cov0[0.9]:.3f}")
        Q_xgb_tuned, c_xgb_tuned, tag = tiny_width_tune(y_va, Q_xgb, TAUS, tau_inv, n_draws=N_DRAWS_BASE, seed=SEED)
        S_xgb = sample_from_quantiles_pchip(Q_xgb_tuned, TAUS, n_draws=N_DRAWS_BASE, seed=SEED, tau_inverse=tau_inv)
        cov_xgb = coverage(y_va, S_xgb)
        log.info(f"XGB tuned ({tag}): CRPS={c_xgb_tuned:.2f} | Cov50/80/90={cov_xgb[0.5]:.3f}/{cov_xgb[0.8]:.3f}/{cov_xgb[0.9]:.3f}")

    # 3) Per-group wrkstat
    with timed("Per-group wrkstat widths"):
        slo_row, shi_row = pergroup_asym_wrkstat(X.loc[X_va.index], y_va, Q_xgb_tuned, TAUS, tau_inv, seed=SEED, min_n=600)
        ts = sorted(Q_xgb_tuned.keys()); t0 = min(ts, key=lambda t: abs(t-0.5)); med = Q_xgb_tuned[t0]
        Q_row = {}
        for t in ts:
            delta = Q_xgb_tuned[t] - med
            scale = np.where(t < t0, slo_row, np.where(t > t0, shi_row, 0.0))
            Q_row[t] = med + scale * delta
        Q_row = enforce_monotone(Q_row)
        S_xgb = sample_from_quantiles_pchip(Q_row, TAUS, n_draws=N_DRAWS_BASE, seed=SEED, tau_inverse=tau_inv)
        c_xgb_row, _ = crps_mean(y_va, S_xgb); cov_xgb_row = coverage(y_va, S_xgb)
        log.info(f"XGB per-group wrkstat: CRPS={c_xgb_row:.2f} | Cov50/80/90={cov_xgb_row[0.5]:.3f}/{cov_xgb_row[0.8]:.3f}/{cov_xgb_row[0.9]:.3f}")

    # 4) Decile-conditional width/bias
    with timed("Decile-conditional width/bias"):
        Q_dec = decile_conditional_tune(y_va, Q_row, TAUS, tau_inv, n_draws=N_DRAWS_BASE, seed=SEED, n_bins=10)
        S_xgb = sample_from_quantiles_pchip(Q_dec, TAUS, n_draws=N_DRAWS_BASE, seed=SEED, tau_inverse=tau_inv)
        c_dec, _ = crps_mean(y_va, S_xgb); cov_dec = coverage(y_va, S_xgb)
        log.info(f"XGB decile-conditional: CRPS={c_dec:.2f} | Cov50/80/90={cov_dec[0.5]:.3f}/{cov_dec[0.8]:.3f}/{cov_dec[0.9]:.3f}")
        Q_row = Q_dec  # carry forward for tail/blend

    # 5) (Optional) NGBoost probes
    with timed("NGBoost Normal@log1p"):
        S_ngb_norm = ngb_samples(X_tr, y_tr, X_va, mode="log1p", Dist=Normal, n_draws=N_DRAWS_BASE, seed=SEED, early_stop=EARLY_STOP)
        c_ngbn, _ = crps_mean(y_va, S_ngb_norm); cov_ngbn = coverage(y_va, S_ngb_norm)
        log.info(f"NGB-N(log1p): CRPS={c_ngbn:.2f} | Cov50/80/90={cov_ngbn[0.5]:.3f}/{cov_ngbn[0.8]:.3f}/{cov_ngbn[0.9]:.3f}")
    with timed("NGBoost LogNormal@raw"):
        S_ngb_logn = ngb_samples(X_tr, y_tr, X_va, mode="raw", Dist=LogNormal, n_draws=N_DRAWS_BASE, seed=1337, early_stop=EARLY_STOP)
        c_ngbl, _ = crps_mean(y_va, S_ngb_logn); cov_ngbl = coverage(y_va, S_ngb_logn)
        log.info(f"NGB-LogN(raw): CRPS={c_ngbl:.2f} | Cov50/80/90={cov_ngbl[0.5]:.3f}/{cov_ngbl[0.8]:.3f}/{cov_ngbl[0.9]:.3f}")

    gap = 100.0  # include NGBoost only if it's >100 better than decile-tuned XGB
    use_ngbn = (c_ngbn + 1e-9) < (c_dec - gap)
    use_ngbl = (c_ngbl + 1e-9) < (c_dec - gap)
    names = ["XGBq"]; cands = []
    # we will build XGB samples after tail autotune (for final); for blend proto use current S_xgb
    cands.append(S_xgb)
    if use_ngbn: names.append("NGB-N(log1p)"); cands.append(S_ngb_norm)
    if use_ngbl: names.append("NGB-LogN(raw)"); cands.append(S_ngb_logn)
    log.info(f"Blend candidates: {names}")

    # 6) Coarse blend (on base draws)
    with timed("Blend search (coarse+CD)"):
        def eval_blend(w):
            S_mix = np.zeros_like(cands[0])
            for wi, Si in zip(w, cands): S_mix += wi * Si
            c, _ = crps_mean(y_va, S_mix); return c
        if len(cands) == 1:
            best_w, best_c = np.array([1.0]), eval_blend(np.array([1.0]))
        else:
            best_w, best_c = None, 1e18
            for w in simplex_grid(len(cands), step=0.25):
                c = eval_blend(w)
                if c < best_c: best_w, best_c = w, c
            best_w, best_c = coordinate_descent(best_w, eval_blend, iters=10, step=0.05)
        log.info(f"Blend best CRPS(base draws)={best_c:.2f} | weights={dict(zip(names, [float(x) for x in best_w]))}")

    # 7) Tail autotune + final resample (deterministic report)
    with timed("Final tail autotune + resample"):
        _, c_tail_probe, (p0_star, p2_star, gam_star) = autotune_tail(Q_row, y_va, TAUS, tau_inv, seed=2025, draws=280)
        log.info(f"Tail autotune (probe): best CRPS={c_tail_probe:.2f} with p0={p0_star:.3f}, p2={p2_star:.3f}, gamma={gam_star:.2f}")

        S_xgb_res = sample_with_gpd_splice(Q_row, TAUS, n_draws=N_DRAWS_FINAL, seed=2025,
                                           tau_inverse=tau_inv, p0=p0_star, p1=0.99, p2=p2_star, gamma=gam_star)
        c_xgb_only, crps_row_xgb = crps_mean(y_va, S_xgb_res)

        if len(cands) == 1:
            S_mix_final, crps_final, crps_row = S_xgb_res, c_xgb_only, crps_row_xgb
            cov_final = coverage(y_va, S_mix_final); w_report = {"XGBq": 1.0}
            log.info(f"FINAL: XGB-only (tuned tail). CRPS={crps_final:.2f} | Cov50/80/90={cov_final[0.5]:.3f}/{cov_final[0.8]:.3f}/{cov_final[0.9]:.3f}")
        else:
            def _resample_cols(S, n_draws, seed):
                rng = _rng(seed); idx = rng.integers(0, S.shape[1], size=n_draws); return S[:, idx]
            S_resampled = [S_xgb_res] + [_resample_cols(Si, N_DRAWS_FINAL, seed=2025) for Si in cands[1:]]
            def eval_blend_on_resampled(w):
                S_mix = np.zeros_like(S_resampled[0])
                for wi, Si in zip(w, S_resampled): S_mix += wi * Si
                c, _ = crps_mean(y_va, S_mix); return c
            w_final, c_blend = coordinate_descent(best_w, eval_blend_on_resampled, iters=8, step=0.05)
            if c_xgb_only <= c_blend:
                S_mix_final, crps_final, crps_row = S_xgb_res, c_xgb_only, crps_row_xgb
                cov_final = coverage(y_va, S_mix_final); w_report = {"XGBq": 1.0}
                log.info(f"FINAL: chose XGB-only (tuned tail). CRPS={crps_final:.2f} | Cov50/80/90={cov_final[0.5]:.3f}/{cov_final[0.8]:.3f}/{cov_final[0.9]:.3f}")
            else:
                S_mix_final = np.zeros_like(S_resampled[0])
                for wi, Si in zip(w_final, S_resampled): S_mix_final += wi * Si
                crps_final, crps_row = crps_mean(y_va, S_mix_final)
                cov_final = coverage(y_va, S_mix_final)
                w_report = dict(zip(names, [float(x) for x in w_final]))
                log.info(f"FINAL: blend wins. CRPS={crps_final:.2f} | Cov50/80/90={cov_final[0.5]:.3f}/{cov_final[0.8]:.3f}/{cov_final[0.9]:.3f}")
                log.info(f"FINAL weights: {w_report}")

    # ---------------- Diagnostics ----------------
    df_crps = pd.DataFrame({"y": y_va.values, "crps_row": crps_row})
    df_crps.sort_values("crps_row", ascending=False).to_csv(ART_DIR / "crps_rows.csv", index=False)

    Q_diag = {float(t): Q_row[t] for t in sorted(Q_row.keys())}
    pd.DataFrame(pinball_asym(y_va, Q_diag, taus=TAUS, k_over=1.25, k_under=1.0)).to_csv(ART_DIR / "pinball_by_tau.csv", index=False)

    ts = np.array(sorted(TAUS)); Qmat = np.vstack([Q_diag[t] for t in ts]).T; yv = y_va.to_numpy()
    idxp = (Qmat <= yv[:, None]).sum(axis=1) - 1; idxp = np.clip(idxp, 0, len(ts) - 2)
    row_idx = np.arange(Qmat.shape[0])
    ql, qh = Qmat[row_idx, idxp], Qmat[row_idx, idxp + 1]; tl, th = ts[idxp], ts[idxp + 1]
    w = np.clip((yv - ql) / np.maximum(qh - ql, 1e-12), 0.0, 1.0); pit = tl + w * (th - tl)
    pd.DataFrame({"pit": pit}).to_csv(ART_DIR / "pit_hist.csv", index=False)
    ks_stat, ks_p = kstest(pit, "uniform", args=(0, 1)); log.info(f"PIT KS stat={ks_stat:.4f}, p={ks_p:.4f}")

    plt.figure(); plt.hist(pit, bins=30, density=True); plt.title("PIT Histogram (final XGB)"); plt.xlabel("u"); plt.ylabel("density")
    plt.tight_layout(); plt.savefig(ART_DIR / "pit_hist.png"); plt.close()

    grid = np.linspace(0, 1, 2001); emp_cdf = np.searchsorted(np.sort(pit), grid, side="right") / len(pit)
    pd.DataFrame({"u": grid, "emp_cdf": emp_cdf}).to_csv(ART_DIR / "calibration_curve.csv", index=False)
    plt.figure(); plt.plot(grid, emp_cdf, label="Empirical CDF(PIT)"); plt.plot(grid, grid, linestyle="--", label="Ideal")
    plt.legend(); plt.title("Calibration Curve (PIT)"); plt.xlabel("u"); plt.ylabel("CDF")
    plt.tight_layout(); plt.savefig(ART_DIR / "calibration_curve.png"); plt.close()

    def _cov(samples, y_true, q):
        lo = np.quantile(samples, (1-q)/2, axis=1); hi = np.quantile(samples, 1-(1-q)/2, axis=1)
        yv2 = np.asarray(y_true, float); return float(np.mean((yv2 >= lo) & (yv2 <= hi)))
    cov_rows = []; Xva_full = X.loc[X_va.index].copy()
    for col in ["wrkstat", "year"]:
        vals = Xva_full[col].astype("category")
        for g in vals.cat.categories:
            mask = (vals.values == g)
            if mask.sum() < 50: continue
            cov50 = _cov(S_mix_final[mask], y_va.values[mask], 0.5)
            cov80 = _cov(S_mix_final[mask], y_va.values[mask], 0.8)
            cov90 = _cov(S_mix_final[mask], y_va.values[mask], 0.9)
            cov_rows.append(dict(group_col=col, group=g, n=int(mask.sum()),
                                 cov50=cov50, cov80=cov80, cov90=cov90))
    pd.DataFrame(cov_rows).to_csv(ART_DIR / "coverage_by_group.csv", index=False)

    med_pred = np.quantile(S_mix_final, 0.5, axis=1)
    bins = np.quantile(med_pred, np.linspace(0, 1, 11)); labels = [f"d{i}" for i in range(1, 11)]
    which = np.digitize(med_pred, bins[1:-1], right=True)
    df_med = pd.DataFrame({"bin": which, "crps_row": crps_row})
    dec = df_med.groupby("bin", dropna=False)["crps_row"].mean().ffill()
    pd.DataFrame({"bin": labels, "mean_crps": dec.values}).to_csv(ART_DIR / "median_bins_crps.csv", index=False)

    summary = dict(
        profile=PROFILE, final_crps=float(crps_final), coverage=cov_final, weights=w_report,
        ks_stat=float(ks_stat), ks_p=float(ks_p), taus=len(TAUS),
        early_stop=EARLY_STOP, n_draws_base=N_DRAWS_BASE, n_draws_final=N_DRAWS_FINAL,
        tuned_xgb_cfg=XGB_CFG_BEST if DO_XGB_TUNE else XGB_FALLBACK
    )
    with open(ART_DIR / "summary.json", "w", encoding="utf-8") as f: json.dump(summary, f, indent=2)
    log.info(f"Artifacts written to: {ART_DIR.resolve()}"); dump_stage_times()

if __name__ == "__main__":
    main()
