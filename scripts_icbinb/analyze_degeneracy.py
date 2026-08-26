#!/usr/bin/env python3
"""Degeneracy audit -> artifacts/results/summaries/DEGENERACY_AUDIT.md.

All three tracked signals are *minimised* by a network that does nothing:
zero activation energy and zero curvature. So on any condition where a
degenerate model happens to score well out-of-distribution -- e.g. because
predicting the training mean is a decent OOD strategy -- every proxy selects it
and appears to reach oracle performance. That is a property of the degeneracy,
not of the loss geometry, and it must be ruled out before any proxy claim.

Two severities are separated because they behave differently:

  constant predictor   pred_std_test < 1e-6.  The network emits one value for
                       every input.  A hard artifact; the condition is unusable
                       for proxy claims without exclusion.
  partially dead       dead_relu_frac > 0.5.  Common and not by itself fatal --
                       such models still discriminate -- but if a selector
                       systematically prefers them it is acting as a
                       dead-unit detector rather than a geometry probe.

The audit therefore reports, per condition: prevalence of both, the dead
fraction *of the model each selector picked*, and every headline contrast
re-run with heavily-dead candidates excluded.
"""
from __future__ import annotations

import glob
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

OUT = "artifacts/results/summaries/DEGENERACY_AUDIT.md"
COLS = ['condition', 'dead_relu_frac', 'dead_relu_frac_penult', 'pred_std_train',
        'pred_std_test', 'test_mse_std', 'val_mse_std', 'train_mse_std',
        'fg_legacy', 'fg_penult', 'hess_top', 'status_train', 'n_params']
SOURCES = [("phase1a_pool", "artifacts/results/phase1a_pool/candidates/*/*.parquet"),
           ("phase1b_seq", "artifacts/results/phase1b_sequential/runs/*/*/*.parquet"),
           ("abl_fixed_arch", "artifacts/results/ablations/fixed_arch/*/*.parquet"),
           ("abl_valfree_pool", "artifacts/results/ablations/valfree_pool/*/*.parquet"),
           ("abl_valfree_seq", "artifacts/results/ablations/valfree_sequential/*/*/*.parquet")]
DEAD = 0.5


def pick(k: pd.DataFrame, s: str | None) -> pd.Series | None:
    if not len(k):
        return None
    if s is None:
        return k.loc[k.val_mse_std.idxmin()]
    if s == "__train":
        return k.loc[k.train_mse_std.idxmin()]
    o = k[k[s].notna()]
    return o.loc[o[s].idxmin()] if len(o) else None


def prevalence(L: list) -> None:
    rows = []
    for name, pat in SOURCES:
        fs = glob.glob(pat)
        if not fs:
            continue
        d = pd.concat([pd.read_parquet(f, columns=COLS) for f in fs])
        d = d[d.status_train == "ok"]
        for cond, g in d.groupby("condition"):
            rows.append((name, cond, len(g), float((g.pred_std_test < 1e-6).mean()),
                         float((g.dead_relu_frac > DEAD).mean()),
                         float(g.dead_relu_frac.median()), float(g.dead_relu_frac.max())))
    rows.sort(key=lambda r: -r[3])
    L.append("\n## 1. Prevalence\n")
    L.append("| experiment | condition | n | constant predictor | dead ReLU >50% | median dead | max dead | verdict |")
    L.append("|---|---|---:|---:|---:|---:|---:|---|")
    for name, cond, n, const, dead, med, mx in rows:
        v = ("**ARTIFACT**" if const > 0.05 else
             "watch" if const > 0.005 or dead > 0.25 else "clean")
        L.append(f"| `{name}` | `{cond}` | {n} | {const:.1%} | {dead:.1%} | "
                 f"{med:.3f} | {mx:.3f} | {v} |")
    L.append("\n`constant predictor` is the hard artifact. A high `dead ReLU >50%` with "
             "0% constant predictors (as on `hydro`) means many units are dead but the "
             "network still discriminates — see section 2 for whether that biases selection.")


def per_condition(L: list) -> None:
    L.append("\n## 2. Does any selector prefer dead networks?\n")
    L.append("`dead@sel` is the median `dead_relu_frac` **of the model that selector chose**. "
             "Compare it against the pool median: a selector well above the pool is acting as "
             "a dead-unit detector.\n")
    targets = [("phase1a_pool", "hydro", "artifacts/results/phase1a_pool/candidates/hydro/*.parquet"),
               ("abl_fixed_arch", "hydro", "artifacts/results/ablations/fixed_arch/hydro/*.parquet"),
               ("phase1a_pool", "amylase", "artifacts/results/phase1a_pool/candidates/amylase/*.parquet"),
               ("phase1a_pool", "cond7", "artifacts/results/phase1a_pool/candidates/cond7/*.parquet")]
    SELS = [("val", None), ("fg_legacy", "fg_legacy"), ("fg_penult", "fg_penult"),
            ("hess_top", "hess_top"), ("train_mse", "__train")]
    for exp, cond, pat in targets:
        fs = sorted(glob.glob(pat))
        if not fs:
            continue
        dfs = [pd.read_parquet(f, columns=COLS) for f in fs]
        base = [g[(g.status_train == "ok") & g.val_mse_std.notna()] for g in dfs]
        pool = pd.concat(base)
        L.append(f"\n### `{exp}` / `{cond}` — {len(dfs)} outer runs "
                 f"(pool median dead = {pool.dead_relu_frac.median():.3f})\n")
        L.append("| selector | dead@sel | median test |")
        L.append("|---|---:|---:|")
        for n, s in SELS:
            r = [pick(k, s) for k in base]
            r = [x for x in r if x is not None]
            L.append(f"| `{n}` | {np.median([x.dead_relu_frac for x in r]):.3f} | "
                     f"{np.median([x.test_mse_std for x in r]):.4f} |")
        L.append(f"\nSpearman: rho(dead, test)={spearmanr(pool.dead_relu_frac, pool.test_mse_std).statistic:+.3f}, "
                 f"rho(dead, fg_legacy)={spearmanr(pool.dead_relu_frac, pool.fg_legacy).statistic:+.3f}, "
                 f"rho(dead, fg_penult)={spearmanr(pool.dead_relu_frac, pool.fg_penult).statistic:+.3f}.")
        # re-run contrasts with heavily-dead candidates removed
        kept = [k[k.dead_relu_frac <= DEAD] for k in base]
        ok = [k for k in kept if len(k) >= 5]
        if len(ok) < 3:
            L.append(f"\n*Exclusion leaves too few candidates ({len(ok)}/{len(dfs)} reps) to re-test.*")
            continue
        b = np.array([pick(k, None).test_mse_std for k in ok])
        L.append(f"\n**Re-tested with `dead_relu_frac > {DEAD}` excluded** "
                 f"({len(ok)}/{len(dfs)} reps keep >=5 candidates; median "
                 f"{np.median([len(k) for k in ok]):.0f}/48 retained):\n")
        L.append("| selector | median test | Δ vs val | better/worse | p |")
        L.append("|---|---:|---:|---:|---:|")
        for n, s in SELS:
            if n == "val":
                continue
            v = np.array([pick(k, s).test_mse_std for k in ok])
            d = v - b
            p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
            L.append(f"| `{n}` | {np.median(v):.4f} | {np.median(d):+.4f} | "
                     f"{int((d < 0).sum())}/{int((d > 0).sum())} | {p:.3f} |")


SWEEP = [("phase1a", "artifacts/results/phase1a_pool/candidates/{c}/*.parquet",
          ["caco2_random", "caco2_scaffold", "lipo_random", "lipo_scaffold",
           "cond7", "amylase", "hydro"]),
         ("fixed_arch", "artifacts/results/ablations/fixed_arch/{c}/*.parquet",
          ["cond7", "hydro"])]
SWEEP_SELS = [("val", None), ("fg_legacy", "fg_legacy"), ("fg_penult", "fg_penult"),
              ("hess_top", "hess_top"), ("val+legacy", "+fg_legacy"),
              ("val+penult", "+fg_penult"), ("train_mse", "__train"), ("n_params", "__np")]


def _sel(k, s):
    if s == "__rand":
        return float(k.test_mse_std.mean())
    if s == "__np":
        return float(k.loc[k.n_params.idxmin(), "test_mse_std"])
    if s is not None and s.startswith("+"):
        c = s[1:]
        o = k[k[c].notna()]
        from scipy.stats import rankdata as _r
        return float(o.iloc[int(np.argmin(_r(o.val_mse_std) + _r(o[c])))].test_mse_std)
    r = pick(k, s)
    return float(r.test_mse_std) if r is not None else np.nan


def _holm(ps):
    n = len(ps)
    a = np.empty(n)
    run = 0.0
    for i, k in enumerate(np.argsort(ps)):
        run = max(run, (n - i) * ps[k])
        a[k] = min(run, 1.0)
    return a


def survives(L: list) -> None:
    """What clears the random-selection bar once degenerate models are removed."""
    L.append("\n## 4. What survives on non-degenerate models only\n")
    L.append("Constant predictors and models with `dead_relu_frac > 0.5` are dropped, then "
             "every selector is tested against **two** baselines: plain `val`, and "
             "`random(E)` — the expected loss of a uniformly random configuration pick. "
             "Holm-corrected within each condition. A selector that beats `val` but not "
             "`random` has not demonstrated signal; it has only shown that validation is "
             "broken on that condition.\n")
    L.append("| experiment / condition | reps | kept | random(E) | val vs random | beats random (Holm<.05) | beats val (Holm<.05) |")
    L.append("|---|---:|---:|---:|---|---|---|")
    for tag, pat, conds in SWEEP:
        for cond in conds:
            fs = sorted(glob.glob(pat.format(c=cond)))
            if not fs:
                continue
            dfs = [pd.read_parquet(f, columns=COLS) for f in fs]
            base = [g[(g.status_train == "ok") & g.val_mse_std.notna()] for g in dfs]
            kept = [k[(k.pred_std_test >= 1e-6) & (k.dead_relu_frac <= DEAD)] for k in base]
            ok = [k for k in kept if len(k) >= 5]
            if len(ok) < 5:
                continue
            rnd = np.array([_sel(k, "__rand") for k in ok])
            v = np.array([_sel(k, None) for k in ok])
            res = {}
            for bname, bvals in [("rand", rnd), ("val", v)]:
                ps, names, meds = [], [], []
                for n, sp in SWEEP_SELS:
                    if bname == "val" and n == "val":
                        continue
                    x = np.array([_sel(k, sp) for k in ok])
                    d = x - bvals
                    d = d[np.isfinite(d)]
                    p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
                    ps.append(p); names.append(n); meds.append(float(np.median(d)))
                adj = _holm(np.array(ps))
                res[bname] = [n for n, m, a in zip(names, meds, adj) if a < 0.05 and m < 0]
            dvr = v - rnd
            pvr = float(wilcoxon(dvr).pvalue) if np.any(dvr != 0) else 1.0
            vr = ("beats random" if (np.median(dvr) < 0 and pvr < 0.05) else
                  "**worse than random**" if (np.median(dvr) > 0 and pvr < 0.05) else
                  "no better than random")
            L.append(f"| `{tag}`/`{cond}` | {len(ok)} | {np.median([len(k) for k in ok]):.0f}/48 | "
                     f"{np.median(rnd):.4f} | {vr} (p={pvr:.3f}) | "
                     f"{', '.join(f'`{x}`' for x in res['rand']) or '—'} | "
                     f"{', '.join(f'`{x}`' for x in res['val']) or '—'} |")
    L.append("\n**Reading this table.** The activation-energy proxies clear the random bar "
             "almost everywhere, so they are not noise — they carry real signal about which "
             "configurations generalise. What they do not do is beat a working validation "
             "set by a margin anyone would act on. Where they appear to beat validation "
             "(`fixed_arch`/`hydro`) it is because validation there is itself worse than "
             "random, and the proxies are only at parity with random.")


def main() -> None:
    L = ["# DEGENERACY AUDIT — ICBINB-BIO 2026", ""]
    L.append("*Auto-generated by `scripts_icbinb/analyze_degeneracy.py`.*")
    L.append("")
    L.append("All three tracked signals are **minimised by a network that does nothing** — a "
             "dead network has zero activation energy and zero curvature. Any condition where "
             "a degenerate model scores well out-of-distribution will therefore show every "
             "proxy 'discovering' it. This audit rules that in or out per condition before "
             "any proxy claim is made.")
    prevalence(L)
    per_condition(L)
    survives(L)
    L.append("\n## 3. Standing conclusions\n")
    L.append("* **`amylase` is unusable for proxy claims.** ~24-28% of models are constant "
             "predictors, and under extreme position shift predicting the training mean is a "
             "strong OOD strategy, so all three signals select the same degenerate model and "
             "appear to reach oracle. Any `amylase` proxy win is an artifact.")
    L.append("* **`hydro` has no constant predictors** but ~30% of models are >50% dead. "
             "Under *variable* architecture the proxies do preferentially select these, so "
             "their small advantage there is partly a dead-unit effect and does not survive "
             "exclusion. Under *fixed* architecture they select **less**-dead models than "
             "validation does, and the advantage survives exclusion — that result is clean.")
    L.append("* **Every other condition is clean** (0% constant predictors, ~2% partially dead).")
    with open(OUT, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {OUT} ({len(L)} lines)")


if __name__ == "__main__":
    main()
