#!/usr/bin/env python3
"""Numbers added to the appendix in response to the ICBINB-BIO reviews.

Reads only stored results; trains nothing. Run from the repository root:

    PYTHONPATH=.:src python scripts_icbinb/review_diagnostics.py

Sections
  1  Confirmatory contrasts (Table 11) with ties and 95% bootstrap intervals
  2  Which audit criterion removes the Amylase proxy advantage
  3  How often each selector deploys a constant predictor
  4  Curvature and proxy values of collapsed networks (lambda_max floor)
  5  Training-mean baseline on the FLIP2 splits
  6  Training loss between epochs 50 and 100
  7  Replications retained by the audit; test- vs training-set constant flag
  8  Confirmatory family with Lipophilicity scaffold added
  9  Tie sizes for the #params selector
 10  NDCG definition that reproduces Table 1

Conventions match the paper: deployment loss is standardised test MSE; a paired
difference is selector minus Val (negative favours the selector); p-values use
scipy.stats.wilcoxon on all paired differences (zeros discarded, normal
approximation whenever a zero is present); Holm is applied across conditions.
"""
from __future__ import annotations
import glob
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata

POOL = "artifacts/results/phase1a_pool/candidates"
PRED = "artifacts/results/phase1a_pool/predictions"
GDSC = "artifacts/results/exploratory/gdsc/phase1a/candidates"
FAMILY = ["cond7", "caco2_scaffold", "amylase", "hydro"]
ALL = ["caco2_random", "caco2_scaffold", "lipo_random", "lipo_scaffold", "cond7",
       "amylase", "hydro"]
N_BOOT, SEED = 4000, 0


# ------------------------------------------------------------------ helpers
def keep(d: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Candidate filters. 'full' is the audit used in the paper."""
    const_te, const_tr = d.pred_std_test < 1e-6, d.pred_std_train < 1e-6
    dead = d.dead_relu_frac > 0.5
    mask = {"none": pd.Series(True, index=d.index),
            "const_test": ~const_te, "const_train": ~const_tr, "dead": ~dead,
            "full": ~const_te & ~dead, "full_train": ~const_tr & ~dead}[rule]
    return d[mask]


def runs(cond: str, rule: str = "none", root: str = POOL) -> list[pd.DataFrame]:
    out = []
    for f in sorted(glob.glob(f"{root}/{cond}/*.parquet")):
        d = pd.read_parquet(f)
        d = d[(d.status_train == "ok") & d.val_mse_std.notna()]
        d = keep(d, rule).reset_index(drop=True)
        if len(d) >= 5:
            out.append(d)
    return out


def pick(k: pd.DataFrame, s: str) -> int:
    """Row index a selector deploys. Ties go to the lowest candidate index."""
    if s == "val":
        return k.val_mse_std.idxmin()
    if s == "train":
        return k.train_mse_std.idxmin()
    if s == "n_params":
        return k.n_params.idxmin()
    if s.startswith("val+"):
        c = s[4:]
        o = k[k[c].notna()]
        return o.index[int(np.argmin(rankdata(o.val_mse_std) + rankdata(o[c])))]
    o = k[k[s].notna()]
    return o[s].idxmin()


def diffs(K, s):
    return np.array([k.loc[pick(k, s), "test_mse_std"] - k.loc[pick(k, "val"), "test_mse_std"]
                     for k in K])


def p_paper(d):
    return float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0


def holm(p):
    p = np.asarray(p, float)
    order, m, run, adj = np.argsort(p), len(p), 0.0, np.empty(len(p))
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def boot_ci(d, rng):
    meds = [np.median(rng.choice(d, len(d))) for _ in range(N_BOOT)]
    return np.percentile(meds, [2.5, 97.5])


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ------------------------------------------------------------------ sections
def s1_confirmatory():
    print("\n## 1  Confirmatory contrasts: median diff, 95% CI, wins/losses/ties, p_Holm")
    for rule in ("none", "full"):
        for s, name in (("fg_legacy", "Proxy"), ("val+fg_legacy", "Val+Proxy")):
            rng = np.random.default_rng(SEED)
            rows, ps = [], []
            for c in FAMILY:
                d = diffs(runs(c, rule), s)
                ps.append(p_paper(d))
                rows.append((c, len(d), np.median(d), *boot_ci(d, rng),
                             (d < 0).sum(), (d > 0).sum(), (d == 0).sum()))
            for (c, n, med, lo, hi, w, l, t), pa in zip(rows, holm(ps)):
                print(f"  {rule:4s} {name:9s} {c:15s} n={n:2d}  {med:+.4f} "
                      f"[{lo:+.4f}, {hi:+.4f}]  {w}/{l}/{t}  p_Holm {fmt_p(pa)}")


def s2_which_criterion():
    print("\n## 2  Proxy vs Val on Amylase under each filter (Holm across the family)")
    for rule in ("none", "const_test", "const_train", "dead", "full"):
        ps, amy = [], None
        for c in FAMILY:
            d = diffs(runs(c, rule), "fg_legacy")
            ps.append(p_paper(d))
            if c == "amylase":
                amy = d
        pa = holm(ps)[FAMILY.index("amylase")]
        print(f"  {rule:11s} median {np.median(amy):+.4f}  w/l {(amy < 0).sum()}/{(amy > 0).sum()}"
              f"  p_Holm {fmt_p(pa)}")
    K = runs("amylase", "const_test")
    sel = [k.loc[pick(k, "fg_legacy")] for k in K]
    print(f"  constants removed (test flag): Proxy's pick has median dead fraction "
          f"{np.median([r.dead_relu_frac for r in sel]):.3f}, median test-prediction std "
          f"{np.median([r.pred_std_test for r in sel]):.1e}")


SELECTORS = [("val", "Val"), ("fg_legacy", "Proxy"), ("fg_penult", "Proxy (pen.)"),
             ("hess_top", "lambda_max"), ("val+fg_legacy", "Val+Proxy"),
             ("val+fg_penult", "Val+Proxy (pen.)"), ("val+hess_top", "Val+lambda_max"),
             ("train", "Train"), ("n_params", "#params")]


def s3_collapse_rates():
    print("\n## 3  Share of unfiltered replications whose deployed model is constant")
    for c in ("amylase", "hydro"):
        K = runs(c)
        print(f"  {c}  ({len(K)} replications)")
        for s, name in SELECTORS:
            r = [k.loc[pick(k, s)] for k in K]
            te = np.mean([x.pred_std_test < 1e-6 for x in r])
            tr = np.mean([x.pred_std_train < 1e-6 for x in r])
            und = sum(not np.isfinite(x.test_spearman) for x in r)
            dead = np.median([x.dead_relu_frac for x in r])
            print(f"    {name:17s} test flag {te:4.0%}  training flag {tr:4.0%}  "
                  f"undefined Spearman {und}/{len(K)}  median dead {dead:.3f}")


def s4_floor():
    print("\n## 4  lambda_max and proxy for constant predictors vs the rest (all conditions)")
    a = pd.concat([pd.concat(runs(c)) for c in ALL])
    c = a.pred_std_test < 1e-6
    print(f"  lambda_max  constant median {a[c].hess_top.median():.2f}   others median "
          f"{a[~c].hess_top.median():.1f}   overall minimum {a.hess_top.min():.4f}")
    print(f"  proxy       constant median {a[c].fg_legacy.median():.1e}   others median "
          f"{a[~c].fg_legacy.median():.3f}")


def s5_train_mean():
    import sys
    sys.path[:0] = [".", "src"]
    from icbinb.datasets import load_flip
    print("\n## 5  Training-mean predictor vs selectors (standardised test MSE, medians)")
    for c in ("amylase", "hydro"):
        y = np.asarray(load_flip(c, 10001).yte, float)   # FLIP2 partitions are fixed
        K = runs(c)
        med = lambda s: np.median([k.loc[pick(k, s), "test_mse_std"] for k in K])
        print(f"  {c}: training mean {np.mean(y ** 2):.3f}   Val {med('val'):.3f}   "
              f"Proxy {med('fg_legacy'):.3f}   oracle "
              f"{np.median([k.test_mse_std.min() for k in K]):.3f}")


def s6_convergence():
    print("\n## 6  Relative fall in training MSE from epoch 50 to epoch 100")
    a = pd.concat([pd.concat(runs(c)) for c in ALL])
    r = (a.train_mse_e50 - a.train_mse_std) / a.train_mse_e50
    q1, q3 = r.quantile([0.25, 0.75])
    print(f"  median {r.median():.1%}  IQR [{q1:.1%}, {q3:.1%}]  "
          f"share falling >10%: {(r > 0.10).mean():.1%}  (n={len(r)})")


def s7_retention():
    print("\n## 7  Replications retained by the audit (>=5 candidates kept)")
    for c in ALL:
        print(f"  {c:15s} {len(runs(c, 'full'))}/{len(runs(c))}   median kept "
              f"{int(np.median([len(k) for k in runs(c, 'full')]))}/48")
    print(f"  gdsc_drug       {len(runs('gdsc_drug', 'full', GDSC))}/{len(runs('gdsc_drug', 'none', GDSC))}")
    roots = [POOL, GDSC, "artifacts/results/exploratory/shift_severity/candidates",
             "artifacts/results/ablations"]
    fs = [f for r in roots for f in glob.glob(f"{r}/**/*.parquet", recursive=True)]
    d = pd.concat([pd.read_parquet(f, columns=["pred_std_test", "pred_std_train",
                                               "dead_relu_frac", "status_train"]) for f in fs])
    d = d[d.status_train == "ok"]
    a = (d.pred_std_test >= 1e-6) & (d.dead_relu_frac <= 0.5)
    b = (d.pred_std_train >= 1e-6) & (d.dead_relu_frac <= 0.5)
    print(f"  constant flag on training instead of test predictions changes the audit "
          f"for {int((a != b).sum())} of {len(d)} candidates")


def s8_family_sensitivity():
    print("\n## 8  Holm family with Lipophilicity scaffold added")
    fam = FAMILY + ["lipo_scaffold"]
    for rule in ("none", "full"):
        for s, name in (("fg_legacy", "Proxy"), ("val+fg_legacy", "Val+Proxy")):
            ps = [p_paper(diffs(runs(c, rule), s)) for c in fam]
            print(f"  {rule:4s} {name:9s} " + "  ".join(
                f"{c} {fmt_p(p)}" for c, p in zip(fam, holm(ps))))


def s9_ties():
    print("\n## 9  Candidates tied at the minimum parameter count, per pool")
    sizes = [int((k.n_params == k.n_params.min()).sum()) for c in ALL for k in runs(c)]
    print(f"  tie sizes observed: {sorted(set(sizes))}")


def s10_ndcg():
    import sys
    sys.path[:0] = [".", "src"]
    from icbinb.datasets import load_flip
    from sklearn.metrics import ndcg_score
    print("\n## 10 NDCG, full deployment set, linear gain = fitness - min fitness, no cutoff")
    for c in ("amylase", "hydro"):
        y = np.asarray(load_flip(c, 10001).yte, float)
        gain = (y - y.min())[None]
        acc = {"Val": [], "Proxy": [], "Random": []}
        for f in sorted(glob.glob(f"{POOL}/{c}/*.parquet")):
            d = pd.read_parquet(f).assign(_row=lambda x: np.arange(len(x)))
            P = np.load(f.replace(POOL, PRED).replace(".parquet", ".npz"))["test_preds"]
            d = keep(d[(d.status_train == "ok") & d.val_mse_std.notna()], "full")
            if len(d) < 5:
                continue
            nd = lambda i: ndcg_score(gain, P[d.loc[i, "_row"]][None])
            acc["Val"].append(nd(d.val_mse_std.idxmin()))
            acc["Proxy"].append(nd(d.fg_legacy.idxmin()))
            acc["Random"].append(np.mean([ndcg_score(gain, P[j][None]) for j in d._row]))
        print(f"  {c}: " + "  ".join(f"{k} {np.mean(v):.3f}" for k, v in acc.items()))


if __name__ == "__main__":
    for fn in (s1_confirmatory, s2_which_criterion, s3_collapse_rates, s4_floor,
               s5_train_mean, s6_convergence, s7_retention, s8_family_sensitivity,
               s9_ties, s10_ndcg):
        fn()
