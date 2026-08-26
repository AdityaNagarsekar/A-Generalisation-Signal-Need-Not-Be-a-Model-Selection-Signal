#!/usr/bin/env python3
"""Phase 1B (sequential HPO) analysis -> artifacts/results/summaries/RESULTS_phase1b.md.

Two questions are kept separate, per plan section 10.1a:

  (a) METHOD-AS-DEFINED: does a proxy-augmented multi-objective search beat a
      val-only search?  Each arm deploys with its own rule, so this confounds
      *search* with *deploy rule* -- it is the practitioner-facing number.
  (b) FREE DECOMPOSITION: every stored trial has val, train and all three
      signals, so any deploy rule can be applied post hoc to any search
      trajectory.  The search x deploy grid separates the two effects at no
      extra compute.

Budget curves use trial_index prefixes (B in {8,16,24,48}); the sampler saw the
trials in that order, so a prefix is exactly the state of the run at budget B.
"""
from __future__ import annotations

import glob
import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

ROOT = "artifacts/results/phase1b_sequential/runs"
OUT = "artifacts/results/summaries/RESULTS_phase1b.md"

SEARCH = ["random_val", "tpe_val", "hebo_val", "tpe_fg_mo", "hebo_fg_mo",
          "tpe_penult_mo", "hebo_penult_mo", "tpe_hess_mo", "hebo_hess_mo"]
DEPLOY = [("val", None), ("val+legacy", "fg_legacy"), ("val+penult", "fg_penult"),
          ("val+hess", "hess_top"), ("train_mse", "__train"), ("oracle", "__oracle")]
# the deploy rule each multi-objective arm uses natively (plan section 10.3)
NATIVE = {"tpe_fg_mo": "val+legacy", "hebo_fg_mo": "val+legacy",
          "tpe_penult_mo": "val+penult", "hebo_penult_mo": "val+penult",
          "tpe_hess_mo": "val+hess", "hebo_hess_mo": "val+hess"}
BUDGETS = [8, 16, 24, 48]


def deploy(g: pd.DataFrame, sig: str | None) -> float:
    """Deployed test loss under one selection rule, over observed trials only."""
    ok = g[(g.status_train == "ok") & g.val_mse_std.notna()]
    if not len(ok):
        return np.nan
    if sig == "__oracle":
        return float(ok.test_mse_std.min())
    if sig == "__train":
        return float(ok.loc[ok.train_mse_std.idxmin(), "test_mse_std"])
    if sig is None:
        return float(ok.loc[ok.val_mse_std.idxmin(), "test_mse_std"])
    o = ok[ok[sig].notna()]
    if not len(o):
        return np.nan
    s = rankdata(o.val_mse_std) + rankdata(o[sig])
    return float(o.iloc[int(np.argmin(s))].test_mse_std)


def holm(ps: np.ndarray) -> np.ndarray:
    n = len(ps)
    adj = np.empty(n)
    run = 0.0
    for i, k in enumerate(np.argsort(ps)):
        run = max(run, (n - i) * ps[k])
        adj[k] = min(run, 1.0)
    return adj


def paired(v: np.ndarray, base: np.ndarray) -> tuple:
    d = v - base
    d = d[np.isfinite(d)]
    p = float(wilcoxon(d).pvalue) if len(d) > 2 and np.any(d != 0) else 1.0
    return float(np.median(d)), int((d < 0).sum()), int((d > 0).sum()), int((d == 0).sum()), p


def load(cond: str) -> dict:
    """{(search, deploy, budget): array over outer runs}"""
    G = {}
    for m in SEARCH:
        files = sorted(glob.glob(f"{ROOT}/{m}/{cond}/*.parquet"))
        dfs = [pd.read_parquet(f) for f in files]
        for B in BUDGETS:
            pref = [d[d.trial_index < B] for d in dfs]
            for dn, sig in DEPLOY:
                G[(m, dn, B)] = np.array([deploy(g, sig) for g in pref])
    return G


def section(cond: str, label: str, L: list) -> None:
    G = load(cond)
    n = len(G[("tpe_val", "val", 48)])
    L.append(f"\n## {cond} — {label}\n")
    L.append(f"{n} outer runs x 9 methods x 48 trials = {n * 9 * 48} trainings **(done)**\n")

    # ---- (a) method as defined, full budget ----
    base = G[("tpe_val", "val", 48)]
    rows, ps = [], []
    for m in SEARCH:
        if m == "tpe_val":
            continue
        v = G[(m, NATIVE.get(m, "val"), 48)]
        md, b, w, t, p = paired(v, base)
        rows.append((m, float(np.median(v)), md, b, w, p))
        ps.append(p)
    adj = holm(np.array(ps))
    L.append("### (a) Method as defined, vs `tpe_val` (B=48)\n")
    L.append("Each arm searches with its own objective and deploys with its own rule.\n")
    L.append("| method | median test | Δ vs tpe_val | better/worse | p | p_holm |")
    L.append("|---|---|---|---|---|---|")
    L.append(f"| tpe_val (reference) | {np.median(base):.4f} | — | — | — | — |")
    for (m, mv, md, b, w, p), a in zip(rows, adj):
        mark = " **\\*\\***" if a < 0.05 else ""
        L.append(f"| {m} | {mv:.4f} | {md:+.4f} | {b}/{w} | {p:.3f} | {a:.3f}{mark} |")

    # ---- (b) search x deploy grid ----
    L.append("\n### (b) Free decomposition — search x deploy grid (median test MSE, B=48)\n")
    L.append("| search \\ deploy | " + " | ".join(d for d, _ in DEPLOY) + " |")
    L.append("|" + "---|" * (len(DEPLOY) + 1))
    for m in SEARCH:
        L.append("| `" + m + "` | " + " | ".join(
            f"{np.median(G[(m, d, 48)]):.4f}" for d, _ in DEPLOY) + " |")

    # ---- deploy-rule main effect, pooled over search ----
    pool_base = np.concatenate([G[(m, "val", 48)] for m in SEARCH])
    rows, ps = [], []
    for dn, _ in DEPLOY:
        if dn == "val":
            continue
        v = np.concatenate([G[(m, dn, 48)] for m in SEARCH])
        md, b, w, t, p = paired(v, pool_base)
        cons = sum(1 for m in SEARCH
                   if np.median(G[(m, dn, 48)]) < np.median(G[(m, "val", 48)]))
        rows.append((dn, float(np.median(v)), md, b, w, t, cons, p))
        ps.append(p)
    adj = holm(np.array(ps))
    L.append(f"\n### (c) Deploy-rule main effect vs `val`, pooled over all 9 searches (n={len(pool_base)} paired)\n")
    L.append("`ties` = the rule selected the identical model validation did.\n")
    L.append("| deploy rule | median | Δ vs val | better | worse | ties | consistent | p | p_holm |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for (dn, mv, md, b, w, t, cons, p), a in zip(rows, adj):
        mark = " **\\*\\***" if a < 0.05 else ""
        L.append(f"| {dn} | {mv:.4f} | {md:+.4f} | {b} | {w} | {t} | {cons}/9 | {p:.2e} | {a:.3f}{mark} |")

    # ---- budget curves ----
    L.append("\n### (d) Budget curves — median deployed test MSE at trial budget B\n")
    L.append("| method | B=8 | B=16 | B=24 | B=48 |")
    L.append("|---|---|---|---|---|")
    for m in SEARCH:
        L.append(f"| `{m}` | " + " | ".join(
            f"{np.median(G[(m, NATIVE.get(m, 'val'), B)]):.4f}" for B in BUDGETS) + " |")
    L.append("| *oracle over pool* | " + " | ".join(
        f"{np.median(np.concatenate([G[(m, 'oracle', B)] for m in SEARCH])):.4f}"
        for B in BUDGETS) + " |")

    # ---- (e) does more search budget help? ----
    rows, ps = [], []
    for grp, name in [(SEARCH[:3], "val-only (3 arms)"),
                      (SEARCH[3:], "proxy-MO (6 arms)"),
                      (SEARCH, "ALL 9 arms")]:
        a = np.concatenate([G[(m, NATIVE.get(m, "val"), 8)] for m in grp])
        b = np.concatenate([G[(m, NATIVE.get(m, "val"), 48)] for m in grp])
        md, better, worse, t, p = paired(b, a)
        rows.append((name, md, better, better + worse + t, p))
        ps.append(p)
    L.append("\n### (e) Does a 6x larger search budget help? (B=8 -> B=48, paired)\n")
    L.append("| arms | median Δ | improved | p |")
    L.append("|---|---|---|---|")
    for nm, md, b, n, p in rows:
        L.append(f"| {nm} | {md:+.4f} | {b}/{n} | {p:.2e} |")


def main() -> None:
    L = ["# RESULTS — ICBINB-BIO 2026 (Phase 1B: sequential HPO)", ""]
    L.append("*Auto-generated by `scripts_icbinb/analyze_phase1b.py`. Protocol locked "
             "and hashed before any of this was inspected.*")
    L.append("")
    L.append("Lower test MSE is better; **negative Δ means the arm beats validation-only "
             "search/selection**. Phase 1A asked whether a proxy can *rank a fixed pool*; "
             "Phase 1B asks the deployable question — whether it can *drive the search*.")
    section("cond7", "**ID-like val -> OOD test** (the hypothesis)", L)
    section("hydro", "**fitness extrapolation** (+4.6 train-sigma)", L)
    L.append("\n## Reading these tables\n")
    L.append("* Panel (a) is the practitioner number but confounds search with deploy rule; "
             "panel (c) isolates the deploy rule at fixed search. Both come from the same runs.")
    L.append("* `train_mse` and `oracle` deploy rules are computable on every trajectory "
             "regardless of what the sampler optimised, so they are free controls.")
    L.append("* Budget curves are prefixes of the *same* runs, not separate shorter runs; "
             "the sampler saw trials in `trial_index` order.")
    L.append("* Holm correction is applied within each panel, within each condition.")
    with open(OUT, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {OUT} ({len(L)} lines)")


if __name__ == "__main__":
    main()
