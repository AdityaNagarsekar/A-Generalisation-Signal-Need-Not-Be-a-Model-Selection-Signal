"""Selectors and per-condition analysis (plan §1, §7.2, §7.3)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata, wilcoxon, binomtest

TEST = "test_mse_std"

# selector -> (column(s), combination). Convention: LOWER score = better model.
SINGLE = {"val":"val_mse_std", "fg_legacy":"fg_legacy", "fg_penult":"fg_penult",
          "hess_top":"hess_top", "train_mse":"train_mse_std",
          "n_params":"n_params", "weight_norm":"weight_norm_fro"}
COMBO  = {"val+legacy":("val_mse_std","fg_legacy"), "val+penult":("val_mse_std","fg_penult"),
          "val+hess":("val_mse_std","hess_top")}

def _eligible(df: pd.DataFrame) -> pd.DataFrame:
    """Plan §4.1: diverged trials are ineligible for selection and excluded from oracle."""
    return df[df.status_train == "ok"]

def select(df: pd.DataFrame, name: str, rng: np.random.Generator | None = None) -> float:
    """Return the TEST loss of the model this selector deploys."""
    d = _eligible(df)
    if len(d) == 0: return float("nan")
    if name == "oracle":  return float(d[TEST].min())
    if name == "random":  return float(d[TEST].iloc[rng.integers(len(d))])
    if name in SINGLE:
        col = SINGLE[name]
        ok = d[d[col].notna()]
        if len(ok) == 0: return float("nan")     # signal failed for every candidate
        return float(ok.loc[ok[col].idxmin(), TEST])
    a, b = COMBO[name]
    ok = d[d[a].notna() & d[b].notna()]
    if len(ok) == 0: return float("nan")
    s = rankdata(ok[a].values) + rankdata(ok[b].values)
    return float(ok.iloc[int(np.argmin(s))][TEST])

def percentile(df: pd.DataFrame, loss: float) -> float:
    d = _eligible(df)
    return float((d[TEST] < loss).mean() * 100)

def rho(df: pd.DataFrame, col: str) -> float:
    d = _eligible(df).dropna(subset=[col, TEST])
    if len(d) < 3 or d[col].std() == 0: return float("nan")
    return float(spearmanr(d[col], d[TEST]).statistic)

def rho_partial(df: pd.DataFrame, col: str) -> float:
    """rho(signal, test | val) by rank residualisation (plan §1 RQ2c)."""
    d = _eligible(df).dropna(subset=[col, TEST, "val_mse_std"])
    if len(d) < 4: return float("nan")
    R = lambda x: rankdata(x)
    v = R(d.val_mse_std.values); s = R(d[col].values); t = R(d[TEST].values)
    resid = lambda y: y - np.polyval(np.polyfit(v, y, 1), v)
    rs, rt = resid(s), resid(t)
    if np.std(rs) == 0 or np.std(rt) == 0: return float("nan")
    return float(spearmanr(rs, rt).statistic)

SELECTORS = ["val","fg_legacy","fg_penult","hess_top","val+legacy","val+penult","val+hess",
             "train_mse","n_params","weight_norm","random","oracle"]

def per_condition(df: pd.DataFrame, seed: int = 0) -> dict:
    """Aggregate one condition over its outer runs."""
    rng = np.random.default_rng(seed)
    per_run = []
    for run, g in df.groupby("outer_run"):
        row = {"outer_run": run}
        for s in SELECTORS:
            row[s] = select(g, s, rng)
        for c in ["val_mse_std","fg_legacy","fg_penult","hess_top","train_mse_std",
                  "n_params","dead_relu_frac"]:
            row[f"rho_{c}"] = rho(g, c)
        for c in ["fg_legacy","fg_penult","hess_top"]:
            row[f"rhopart_{c}"] = rho_partial(g, c)
        row["pct_val"] = percentile(g, row["val"])
        row["pct_fg"]  = percentile(g, row["fg_legacy"])
        per_run.append(row)
    P = pd.DataFrame(per_run)
    out = {"n_runs": len(P), "per_run": P, "deploy": {}, "contrast": {}}
    for s in SELECTORS:
        out["deploy"][s] = (P[s].median(), P[s].mean(), P[s].std())
    base = P["val"].values
    for s in SELECTORS:
        if s == "val": continue
        d = P[s].values - base
        d = d[np.isfinite(d)]
        if len(d) < 3 or np.allclose(d, 0):
            out["contrast"][s] = (np.nan, np.nan, np.nan, 0); continue
        nz = d[d != 0]
        p = float(wilcoxon(nz).pvalue) if len(nz) else 1.0
        out["contrast"][s] = (float(np.median(d)), float(np.mean(d)), p, int((d < 0).sum()))
    return out
