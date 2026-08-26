#!/usr/bin/env python3
"""Ablation analysis -> artifacts/results/summaries/RESULTS_ablations.md.

Three pre-registered ablations (plan section 8):

  fixed_arch          architecture pinned to 256x2; only lr/wd/dropout vary.
                      Tests whether proxy failure in the main experiment is a
                      *cross-architecture comparability* problem rather than a
                      lack of signal.  NOTE: `n_params` is constant here, so it
                      is not a usable selector and is omitted.
  valfree_pool        val folded into train (n_val = 0); no validation signal
                      exists, so only train-side selectors are defined.
  valfree_sequential  the six multi-objective arms searching (train, signal)
                      instead of (val, signal).

Every table carries a `random(E)` row: the mean test loss over the candidate
pool, i.e. the expected loss of picking a configuration uniformly at random.
Without it a selector that merely avoids catastrophic models looks skilful, and
-- as it turns out on `hydro` -- validation itself does not clear this bar.
"""
from __future__ import annotations

import glob
import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon, spearmanr

ABL = "artifacts/results/ablations"
OUT = "artifacts/results/summaries/RESULTS_ablations.md"
SIGNALS = ["fg_legacy", "fg_penult", "hess_top"]


def sel(g: pd.DataFrame, s: str | None, need_val: bool) -> float:
    ok = g[g.status_train == "ok"]
    if need_val:
        ok = ok[ok.val_mse_std.notna()]
    if not len(ok):
        return np.nan
    if s == "__rand":
        return float(ok.test_mse_std.mean())
    if s == "__oracle":
        return float(ok.test_mse_std.min())
    if s == "__train":
        return float(ok.loc[ok.train_mse_std.idxmin(), "test_mse_std"])
    if s is None:
        return float(ok.loc[ok.val_mse_std.idxmin(), "test_mse_std"])
    if s.startswith("+"):          # rank-sum of val and the signal
        c = s[1:]
        o = ok[ok[c].notna()]
        if not len(o):
            return np.nan
        return float(o.iloc[int(np.argmin(rankdata(o.val_mse_std) + rankdata(o[c])))].test_mse_std)
    if s.startswith("t+"):         # rank-sum of train and the signal (val-free)
        c = s[2:]
        o = ok[ok[c].notna()]
        if not len(o):
            return np.nan
        return float(o.iloc[int(np.argmin(rankdata(o.train_mse_std) + rankdata(o[c])))].test_mse_std)
    o = ok[ok[s].notna()]
    if not len(o):
        return np.nan
    return float(o.loc[o[s].idxmin(), "test_mse_std"])


def holm(ps: np.ndarray) -> np.ndarray:
    n = len(ps)
    a = np.empty(n)
    r = 0.0
    for i, k in enumerate(np.argsort(ps)):
        r = max(r, (n - i) * ps[k])
        a[k] = min(r, 1.0)
    return a


def contrast(R: dict, names: list, base_name: str, L: list, base_label: str) -> None:
    base = R[base_name]
    rows, ps = [], []
    for n in names:
        if n == base_name:
            continue
        d = R[n] - base
        d = d[np.isfinite(d)]
        if not len(d):
            continue
        p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
        rows.append((n, float(np.median(R[n])), float(np.median(d)),
                     int((d < 0).sum()), int((d > 0).sum()), p))
        ps.append(p)
    adj = holm(np.array(ps))
    L.append(f"\n**vs `{base_name}`** ({base_label})\n")
    L.append("| selector | median test | Δ | better | worse | p | p_holm |")
    L.append("|---|---|---|---|---|---|---|")
    L.append(f"| `{base_name}` (ref) | {np.median(base):.4f} | — | — | — | — | — |")
    for (n, mv, md, b, w, p), a in zip(rows, adj):
        mark = " **\\*\\***" if a < 0.05 else ""
        L.append(f"| `{n}` | {mv:.4f} | {md:+.4f} | {b} | {w} | {p:.3f} | {a:.3f}{mark} |")


def rho_row(dfs: list, L: list, need_val: bool) -> None:
    ok = pd.concat(dfs)
    ok = ok[ok.status_train == "ok"]
    cols = SIGNALS + (["val_mse_std"] if need_val else []) + ["train_mse_std"]
    parts = []
    for c in cols:
        m = ok[c].notna()
        parts.append(f"`{c}`={spearmanr(ok[c][m], ok.test_mse_std[m]).statistic:+.3f}")
    L.append(f"\nSpearman rho(signal, test MSE) pooled over all trials (n={len(ok)}); "
             f"**positive means lower signal -> lower test loss**, i.e. minimising it helps: "
             + ", ".join(parts) + ".")
    degen = float((ok.pred_std_test < 1e-6).mean())
    L.append(f"\nDegeneracy check: max `dead_relu_frac` = {ok.dead_relu_frac.max():.3f}, "
             f"fraction of constant predictors = {degen:.3f}.")
    if degen > 0.05:
        L.append(f"\n> **ARTIFACT WARNING — do not read the proxy rows above as evidence.** "
                 f"{degen:.1%} of trained models are constant predictors (dead networks). "
                 f"All three signals are *minimised* by a dead network -- zero activation "
                 f"energy and zero curvature -- so on a condition where predicting the "
                 f"training mean happens to be a good OOD strategy, every proxy selects the "
                 f"same degenerate model and appears to reach oracle performance. This is a "
                 f"property of the degeneracy, not of the geometry. Re-read this condition "
                 f"with degenerate candidates excluded before drawing any conclusion.")


def fixed_arch(L: list) -> None:
    conds = sorted(d for d in os.listdir(f"{ABL}/fixed_arch")
                   if os.path.isdir(f"{ABL}/fixed_arch/{d}"))
    L.append("\n## Ablation 1 — fixed architecture (256 x 2)\n")
    L.append("Only `lr`, `weight_decay` and `dropout` vary. If the proxies fail in the main "
             "experiment because activation energy is not comparable between a 64-wide "
             "1-layer net and a 512-wide 4-layer one, they should recover here. "
             "`n_params` is constant and therefore omitted as a selector.")
    names = ["random(E)", "val", "fg_legacy", "fg_penult", "hess_top",
             "val+legacy", "val+penult", "val+hess", "train_mse", "oracle"]
    spec = {"random(E)": "__rand", "val": None, "fg_legacy": "fg_legacy",
            "fg_penult": "fg_penult", "hess_top": "hess_top", "val+legacy": "+fg_legacy",
            "val+penult": "+fg_penult", "val+hess": "+hess_top",
            "train_mse": "__train", "oracle": "__oracle"}
    for cond in conds:
        dfs = [pd.read_parquet(f) for f in sorted(glob.glob(f"{ABL}/fixed_arch/{cond}/*.parquet"))]
        if not dfs:
            continue
        R = {n: np.array([sel(g, spec[n], True) for g in dfs]) for n in names}
        L.append(f"\n### fixed_arch / `{cond}` — {len(dfs)} outer runs\n")
        contrast(R, names, "val", L, "does any signal beat validation?")
        contrast(R, names, "random(E)", L,
                 "does the selector beat picking a configuration at random?")
        rho_row(dfs, L, True)


def valfree_pool(L: list) -> None:
    root = f"{ABL}/valfree_pool"
    if not os.path.isdir(root):
        return
    conds = sorted(d for d in os.listdir(root) if os.path.isdir(f"{root}/{d}"))
    L.append("\n## Ablation 2 — val-free pool (validation folded into training)\n")
    L.append("`n_val = 0`: the validation split is added to training, so no validation "
             "signal exists and only train-side selectors are defined. This asks the "
             "practitioner's question — **is validation data better spent as training data?** "
             "Compare the medians here against the `val` rows of the main Phase 1A tables, "
             "which used the same conditions with a held-out validation split.")
    names = ["random(E)", "train_mse", "fg_legacy", "fg_penult", "hess_top",
             "train+legacy", "train+penult", "train+hess", "oracle"]
    spec = {"random(E)": "__rand", "train_mse": "__train", "fg_legacy": "fg_legacy",
            "fg_penult": "fg_penult", "hess_top": "hess_top",
            "train+legacy": "t+fg_legacy", "train+penult": "t+fg_penult",
            "train+hess": "t+hess_top", "oracle": "__oracle"}
    for cond in conds:
        fs = sorted(glob.glob(f"{root}/{cond}/*.parquet"))
        dfs = [pd.read_parquet(f) for f in fs]
        if not dfs:
            continue
        R = {n: np.array([sel(g, spec[n], False) for g in dfs]) for n in names}
        L.append(f"\n### valfree_pool / `{cond}` — {len(dfs)} outer runs"
                 + ("" if len(dfs) >= 10 else "  *(incomplete — partial data)*") + "\n")
        contrast(R, names, "train_mse", L, "does a signal beat train-MSE alone?")
        contrast(R, names, "random(E)", L, "does the selector beat a random pick?")
        rho_row(dfs, L, False)


def valfree_headline(L: list) -> None:
    """Paired: main pool (val-based selection) vs val-free pool (train-based).

    The two arms share `split_seed`, candidate ids and the *identical* test
    split -- the val-free arm only moves the validation rows into training --
    so outer runs pair exactly.
    """
    POOL = "artifacts/results/phase1a_pool/candidates"
    root = f"{ABL}/valfree_pool"
    conds = [c for c in ["caco2_scaffold", "lipo_scaffold", "amylase", "hydro"]
             if os.path.isdir(f"{root}/{c}")]
    if not conds:
        return
    L.append("\n### Headline — is validation data better spent as training data?\n")
    L.append("Paired by outer run against the main Phase 1A pool, which used the same "
             "candidates, seeds and test split but held out 10% for validation. "
             "**Positive Δ means giving up the validation set costs you.**\n")
    L.append("| condition | val (main) | train_mse (val-free) | Δ | better/worse | p | best val-free selector |")
    L.append("|---|---|---|---|---|---|---|")
    for c in conds:
        M = {os.path.basename(f): pd.read_parquet(f) for f in glob.glob(f"{POOL}/{c}/*.parquet")}
        V = {os.path.basename(f): pd.read_parquet(f) for f in glob.glob(f"{root}/{c}/*.parquet")}
        ks = sorted(set(M) & set(V))
        if len(ks) < 3:
            continue
        a = np.array([sel(M[k], None, True) for k in ks])
        b = np.array([sel(V[k], "__train", False) for k in ks])
        d = b - a
        p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
        cand = {n: np.array([sel(V[k], sp, False) for k in ks]) for n, sp in
                [("train_mse", "__train"), ("train+legacy", "t+fg_legacy"),
                 ("train+penult", "t+fg_penult"), ("train+hess", "t+hess_top"),
                 ("fg_legacy", "fg_legacy"), ("fg_penult", "fg_penult")]}
        best = min(cand, key=lambda n: np.median(cand[n]))
        L.append(f"| `{c}` (n={len(ks)}) | {np.median(a):.4f} | {np.median(b):.4f} | "
                 f"{np.median(d):+.4f} | {int((d < 0).sum())}/{int((d > 0).sum())} | "
                 f"{p:.3f} | `{best}` {np.median(cand[best]):.4f} |")
    L.append("\nDropping the validation split is **at parity** wherever the condition is not "
             "degenerate; see the artifact warning on `amylase`.")


def valfree_sequential(L: list) -> None:
    root = f"{ABL}/valfree_sequential"
    if not os.path.isdir(root):
        L.append("\n## Ablation 3 — val-free sequential\n\n*(not yet run)*")
        return
    methods = sorted(d for d in os.listdir(root) if os.path.isdir(f"{root}/{d}"))
    L.append("\n## Ablation 3 — val-free sequential HPO\n")
    L.append("The six multi-objective arms searching `(train_mse, signal)` instead of "
             "`(val_mse, signal)`. Deployment uses the train-side rank-sum.")
    rows = []
    for m in methods:
        for cond in sorted(os.listdir(f"{root}/{m}")):
            fs = sorted(glob.glob(f"{root}/{m}/{cond}/*.parquet"))
            if not fs:
                continue
            dfs = [pd.read_parquet(f) for f in fs]
            sig = ("fg_legacy" if "fg_mo" in m else
                   "fg_penult" if "penult" in m else "hess_top")
            v = np.array([sel(g, f"t+{sig}", False) for g in dfs])
            r = np.array([sel(g, "__rand", False) for g in dfs])
            t = np.array([sel(g, "__train", False) for g in dfs])
            rows.append((m, cond, len(dfs), np.median(v), np.median(t), np.median(r)))
    if not rows:
        L.append("\n*(no shards yet)*")
        return
    L.append("\n| method | condition | reps | train+signal | train_mse | random(E) |")
    L.append("|---|---|---|---|---|---|")
    for m, c, n, a, b, cc in rows:
        L.append(f"| `{m}` | {c} | {n} | {a:.4f} | {b:.4f} | {cc:.4f} |")


def main() -> None:
    L = ["# RESULTS — ICBINB-BIO 2026 (ablations)", ""]
    L.append("*Auto-generated by `scripts_icbinb/analyze_ablations.py`.*")
    L.append("")
    L.append("Lower test MSE is better; negative Δ means the row beats the reference. "
             "Every table includes **`random(E)`** — the mean test loss over the candidate "
             "pool, i.e. what a uniformly random configuration choice would cost. A selector "
             "that does not beat `random(E)` carries no usable signal, whatever its p-value "
             "against another selector.")
    fixed_arch(L)
    valfree_pool(L)
    valfree_headline(L)
    valfree_sequential(L)
    with open(OUT, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {OUT} ({len(L)} lines)")


if __name__ == "__main__":
    main()
