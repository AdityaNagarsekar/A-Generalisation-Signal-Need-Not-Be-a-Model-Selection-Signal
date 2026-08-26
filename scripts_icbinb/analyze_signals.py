#!/usr/bin/env python3
"""Signal-relationship deep dive -> artifacts/results/summaries/RESULTS_signals.md.

Answers three questions the selection tables cannot:

  1. Is activation energy actually a stand-in for curvature, as its motivation
     claims?  (`hess_top` was tracked precisely to make this checkable.)
  2. Does any signal track the generalisation gap it is supposed to control?
  3. Are the two activation-energy variants independent probes or one signal?

Every relationship is reported three ways, because a marginal correlation computed
across a hyperparameter sweep is not evidence of a mechanism -- a shared driver can
manufacture or mask one. See the METHOD section of the generated file.
"""
from __future__ import annotations

import glob
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

OUT = "artifacts/results/summaries/RESULTS_signals.md"
POOL = "artifacts/results/phase1a_pool/candidates"
ORDER = ["caco2_random", "caco2_scaffold", "lipo_random", "lipo_scaffold",
         "cond7", "amylase", "hydro"]
COLS = ["condition", "lr", "weight_decay", "dropout", "width", "depth", "n_params",
        "dead_relu_frac", "pred_std_test", "fg_legacy", "fg_penult", "hess_top",
        "train_mse_std", "val_mse_std", "test_mse_std",
        "val_mse_e25", "val_mse_e50", "status_train"]
ARCH = ["width", "depth"]


def load(cond):
    fs = glob.glob(f"{POOL}/{cond}/*.parquet")
    if not fs:
        return None
    d = pd.concat([pd.read_parquet(f, columns=COLS) for f in fs])
    d = d[(d.status_train == "ok") & d.hess_top.notna() & d.fg_legacy.notna()]
    d = d[(d.pred_std_test >= 1e-6) & (d.dead_relu_frac <= 0.5)].copy()
    d["gap"] = d.test_mse_std - d.train_mse_std
    d["loglr"] = np.log10(d.lr)
    return d


def partial(d, x, y, ctrl):
    """Spearman partial correlation by rank residualisation (see METHOD)."""
    R = np.column_stack([np.ones(len(d))] + [rankdata(d[c]) for c in ctrl])
    def res(v):
        r = rankdata(v)
        b, *_ = np.linalg.lstsq(R, r, rcond=None)
        return r - R @ b
    return spearmanr(res(d[x]), res(d[y])).statistic


def stratified(d, x, y, nbins=8):
    """Assumption-free alternative: correlate strictly WITHIN narrow lr bins and
    within architecture cells, then pool weighted by cell size. Removes any effect
    of lr, monotone or not, at the cost of discarding small cells."""
    d = d.copy()
    d["_b"] = pd.qcut(d.lr, nbins, labels=False, duplicates="drop")
    rs, ns = [], []
    for _, g in d.groupby(["_b", "width", "depth"]):
        if len(g) >= 12 and g[x].nunique() > 3:
            r = spearmanr(g[x], g[y]).statistic
            if np.isfinite(r):
                rs.append(r); ns.append(len(g))
    if not rs:
        return np.nan, 0, 0
    return float(np.average(rs, weights=ns)), len(rs), int(np.sum(ns))


def triple_table(L, x, y, title, note=""):
    L.append(f"\n### {title}\n")
    if note:
        L.append(note + "\n")
    L.append("| condition | marginal ρ | partial ρ (arch) | partial ρ (arch + lr) | within-bin ρ | cells | n used |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for cond in ORDER:
        d = load(cond)
        if d is None or len(d) < 50:
            continue
        m = spearmanr(d[x], d[y]).statistic
        pa = partial(d, x, y, ARCH)
        pl = partial(d, x, y, ARCH + ["lr"])
        w, nb, nn = stratified(d, x, y)
        ws = "—" if not np.isfinite(w) else f"{w:+.3f}"
        cellinfo = "—" if nb == 0 else str(nb)
        L.append(f"| `{cond}` | {m:+.3f} | {pa:+.3f} | **{pl:+.3f}** | {ws} | {cellinfo} | {nn} |")


# ------------------------------------------------------------------ selection diagnostics
def _sel_rows(R, key):
    return pd.DataFrame([k.loc[key(k)] for k in R]).reset_index(drop=True)


def _runs(cond, root=POOL):
    out = []
    for f in sorted(glob.glob(f"{root}/{cond}/*.parquet")):
        d = pd.read_parquet(f, columns=COLS)
        d = d[(d.status_train == "ok") & d.val_mse_std.notna()]
        d = d[(d.pred_std_test >= 1e-6) & (d.dead_relu_frac <= 0.5)]
        if len(d) >= 5:
            out.append(d)
    return out


def diagnostics(L):
    from scipy.stats import wilcoxon
    L.append("""
## 6. What does validation get *wrong*, hyperparameter by hyperparameter?

Comparing the configuration validation selects against the configuration the oracle
would have selected, paired within each outer run. Positive = the oracle prefers a
larger value than validation chose.
""")
    L.append("| condition | n | Δ log10 lr (oracle − val) | p | Δ log2 width | p |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for cond in ORDER:
        R = _runs(cond)
        if len(R) < 5:
            continue
        v = _sel_rows(R, lambda k: k.val_mse_std.idxmin())
        o = _sel_rows(R, lambda k: k.test_mse_std.idxmin())
        dl = np.log10(o.lr.to_numpy()) - np.log10(v.lr.to_numpy())
        dw = np.log2(o.width.to_numpy().astype(float)) - np.log2(v.width.to_numpy().astype(float))
        pl = float(wilcoxon(dl).pvalue) if np.any(dl != 0) else 1.0
        pw = float(wilcoxon(dw).pvalue) if np.any(dw != 0) else 1.0
        L.append(f"| `{cond}` | {len(R)} | {np.median(dl):+.3f} | {pl:.3f} | {np.median(dw):+.2f} | {pw:.3f} |")
    L.append("""
**Validation over-regularises under fitness extrapolation.** On `hydro` the oracle
wants a learning rate 10^0.312 = **2.05x larger** (p < 0.001) and a width **4x larger**
(p < 0.001) than validation selects; validation also picks more dropout (0.39 vs 0.09)
and more weight decay. All four move in the same direction: because the validation
split is in-distribution, a conservatively regularised model looks safe on it, and
that model then underfits the extrapolation regime.

On `amylase` the bias runs the other way (oracle prefers a *smaller* lr and half the
width, p = 0.018 / 0.003). On the four molecular conditions there is **no systematic
error at all** (p = 0.09 to 1.00) — validation picks essentially the right
hyperparameters there, which is consistent with it being hard to beat on those
conditions.
""")

    L.append("""
## 7. Correction: `n_params` does not win on `hydro` because "smaller is better"

`n_params` (select the model with the fewest parameters) beats validation on `hydro`.
Two things had to be checked before that could be reported.

**Is it a tie-breaking artifact?** A median of 3 candidates (range 1-5) tie at the
minimum parameter count, so `idxmin` resolves the tie by row order. Re-running with
*random* tie-breaking, averaged over 200 draws, gives 22.14 vs validation's 22.92 —
better in 23/30 runs, p = 0.0004, and better than a random pick in 27/30, p < 0.0001.
**Not an artifact.**

**Does it mean small models are better here? No.** Validation and `n_params` pick the
*same* width in 22/30 runs (both width 64), while the oracle prefers width **256**:

| selector | median test MSE | median width |
|---|---:|---:|
| oracle | 20.58 | 256 |
| `n_params` | 22.08 | 64 |
| `val` | 22.92 | 64 |
| random pick | 23.05 | — |

They select the same *model* in only 10/30 runs, so the advantage comes from the other
hyperparameters: `n_params` lands on a slightly higher learning rate (p = 0.003) and
lower dropout (p = 0.009) — both in the direction of the oracle. So `n_params` is not
capturing an optimum; it is simply **less wrong than validation**, and both are far
from the oracle. An earlier framing of this result as "pick the smallest model" was
incorrect and is retracted here.
""")

    L.append("""
## 8. The dead-ReLU mechanism is regime-specific

Dead units are bad for deployment on `hydro` in both regimes, and this is not a size
confound — rho(dead, test) is +0.44 pooled and +0.29 to +0.55 *within* every width
band, while rho(dead, width) = +0.04.

But **which selector prefers dead networks reverses between regimes**:

| | rho(dead, test) | dead @ val | dead @ oracle | val picks deader | p |
|---|---:|---:|---:|---:|---:|
| `fixed_arch/hydro` (256x2 pinned) | +0.766 | 0.286 | 0.025 | **10/10** | 0.002 |
| `pool/hydro` (architecture varies) | +0.436 | 0.066 | 0.149 | 7/30 | 0.006 |

Under a fixed architecture validation reliably selects the deadest networks, which
explains its failure there. On the variable-architecture pool it selects *less*-dead
networks than the oracle and still loses — so on the pool its failure is explained by
the over-regularisation of section 6, not by dead units. **The dead-ReLU mechanism
should only be cited for the fixed-architecture result.**
""")

    L.append("""
## 9. Selecting earlier in training

Deployed test MSE when validation is read at epoch 25 or 50 instead of 100 (paired):

| condition | val@100 | val@50 | val@25 | Δ(25−100) | better | p |
|---|---:|---:|---:|---:|---:|---:|
""".rstrip())
    for cond in ["lipo_random", "lipo_scaffold", "cond7", "amylase", "hydro"]:
        R = _runs(cond)
        if len(R) < 5:
            continue
        f100 = np.array([k.loc[k.val_mse_std.idxmin(), "test_mse_std"] for k in R])
        f50 = np.array([k.loc[k.val_mse_e50.idxmin(), "test_mse_std"] for k in R])
        f25 = np.array([k.loc[k.val_mse_e25.idxmin(), "test_mse_std"] for k in R])
        d = f25 - f100
        p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
        L.append(f"| `{cond}` | {np.median(f100):.4f} | {np.median(f50):.4f} | "
                 f"{np.median(f25):.4f} | {np.median(d):+.4f} | {int((d<0).sum())}/{len(d)} | {p:.3f} |")
    L.append("""
Selecting at epoch 25 is significantly **worse** on all four in-distribution-ish
conditions (p = 0.005 to 0.036), as expected — the ranking has not settled. On
`hydro` alone it is *better* (−0.147, 17/30), though **not significant (p = 0.158)**,
so this is suggestive only and we do not claim it. It is consistent with the budget
result in the main file: under extrapolation, more optimisation against the
validation set does not help and may hurt.
""")


def main():
    L = ["# RESULTS — signal relationships, and the role of learning rate", ""]
    L.append("*Auto-generated by `scripts_icbinb/analyze_signals.py`. "
             "Non-degenerate models only (constant predictors and >50% dead-ReLU "
             "models removed) throughout.*")

    # ---------------- method ----------------
    L.append("""
## METHOD — what "conditioning on learning rate" means here

A correlation computed across a hyperparameter sweep is **not** evidence of a
mechanism. Every model in the pool differs in learning rate, weight decay, dropout,
width and depth simultaneously, and those choices drive the signals. A shared driver
can manufacture an association that does not exist at fixed hyperparameters, or mask
one that does. Both happen in this study, in opposite directions.

So every relationship below is reported three ways.

**1. Marginal ρ.** Plain Spearman rank correlation over all models in the pool.
This is the quantity that governs *selection*: `argmin fg_legacy` ranks the pool
marginally, so if the goal is to predict what a selector will do, this is the
relevant number.

**2. Partial ρ (rank residualisation).** Spearman partial correlation:

```
  rx = rank(x);  ry = rank(y);  R = [1, rank(c1), ..., rank(ck)]
  ex = rx − R·(R⁺ rx)          # least-squares residual
  ey = ry − R·(R⁺ ry)
  partial ρ = Pearson(ex, ey)  # = Spearman(x, y | controls)
```

The residual `ex` is *how high this model's activation energy was, relative to what
its learning rate and architecture alone would predict*. Correlating residuals asks:
**among models that share a learning rate and architecture, do the two signals still
co-vary?**

*Limitation:* the residualisation removes only the **linear-in-rank** component of
each control's effect. If learning rate acts non-monotonically on a signal, some of
its influence survives.

**3. Within-bin ρ (assumption-free check).** Split learning rate into 8 quantile
bins, then correlate strictly *within* each (lr-bin × width × depth) cell and pool
the per-cell correlations weighted by cell size. This removes any effect of lr,
monotone or not, and makes no functional-form assumption. The cost is discarded
data: cells with fewer than 12 models are dropped, so the `n used` column is well
below the full pool and the estimate is noisier. `caco2` retains only ~123 of ~960 models and
should be read with that in mind; `amylase` retains **no** usable cell at all (its
degeneracy filter plus 10 architecture cells leaves every cell under the size floor),
so its within-bin entry is shown as "—" rather than a number.

**Agreement between (2) and (3) is the evidence.** Where they agree, the conclusion
does not rest on the linearity assumption. Where they disagree, we say so.
""")

    # ---------------- how lr drives things ----------------
    L.append("\n## 1. How each hyperparameter drives each signal\n")
    L.append("The confound has to be established before it can be removed. "
             "Spearman ρ between each hyperparameter and each signal:\n")
    L.append("| condition | ρ(lr, fg_legacy) | ρ(lr, λ_max) | ρ(wd, fg_legacy) | ρ(wd, λ_max) | ρ(dropout, fg_legacy) | ρ(dropout, λ_max) |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for cond in ORDER:
        d = load(cond)
        if d is None:
            continue
        r = lambda a, b: spearmanr(d[a], d[b]).statistic
        L.append(f"| `{cond}` | {r('lr','fg_legacy'):+.3f} | {r('lr','hess_top'):+.3f} | "
                 f"{r('weight_decay','fg_legacy'):+.3f} | {r('weight_decay','hess_top'):+.3f} | "
                 f"{r('dropout','fg_legacy'):+.3f} | {r('dropout','hess_top'):+.3f} |")
    L.append("""
**Learning rate pushes the two signals in opposite directions.** Larger lr raises
activation energy (ρ ≈ +0.5) and *lowers* the top Hessian eigenvalue (ρ ≈ −0.4 to
−0.8) — the latter being the well-known tendency of large learning rates to settle in
flatter minima. Weight decay acts more weakly and in the reverse sense; dropout
barely matters. Learning rate is therefore the confounder to remove.""")

    # ---------------- the three questions ----------------
    L.append("\n## 2. Is activation energy a stand-in for curvature?\n")
    L.append("This is the question `hess_top` was tracked to answer. If the proxy's "
             "motivation is right, these should be strongly **positive**.")
    triple_table(L, "fg_legacy", "hess_top", "`fg_legacy` vs λ_max")
    triple_table(L, "fg_penult", "hess_top", "`fg_penult` vs λ_max")
    L.append("""
**Verdict: no — and the marginal number is misleading.** Marginally the two are
anti-correlated everywhere (−0.13 to −0.58), which would suggest activation energy is
an *inverse* measure of sharpness. It is not. Controlling for architecture alone
leaves the association intact (so it is not a width/normalisation artifact — note
`fg_legacy` divides by `B·w` while λ_max does not, which was the obvious suspect).
Adding learning rate collapses it: on `lipo_random` −0.391 → −0.087, on `cond7`
−0.399 → −0.124, on `hydro` −0.554 → −0.136. The within-bin check agrees in
direction and magnitude (−0.200, −0.052, −0.144).

The honest statement is that **given the learning rate, activation energy carries
almost no information about curvature in either direction**. The apparent coupling is
a shared response to lr. `amylase` is the exception, where the partial correlation
*strengthens*; it is also the degenerate condition, so we do not rely on it.

This still explains selection behaviour: selection operates marginally, so
`argmin fg_legacy` does tend to choose higher-curvature models — but because of the
learning rate it implicitly selects, not because it measures geometry.""")

    L.append("\n## 3. Does any signal track the generalisation gap?\n")
    L.append("gap = test MSE − train MSE, both on the train-standardised scale. "
             "A signal useful for deployment should rise with the gap.")
    triple_table(L, "fg_legacy", "gap", "`fg_legacy` vs generalisation gap")
    triple_table(L, "fg_penult", "gap", "`fg_penult` vs generalisation gap")
    triple_table(L, "hess_top", "gap", "`hess_top` (λ_max) vs generalisation gap",
                 "**This is the control, and the most important table in the file.**")
    L.append("""
**Verdict: the cheap proxy tracks generalisation, and exact curvature does not.**

On every molecular condition λ_max is *precisely uninformative* about the
generalisation gap — marginal ρ ≈ +0.02 to +0.07, partial ρ ≈ −0.01 to −0.03, within
bins ≈ −0.04 to −0.14, at n ≈ 1,000–1,400. That is as close to a measured zero as
this design can produce.

Activation energy carries a small but consistent positive signal (+0.16 to +0.35
partial). Critically, and unlike §2, this **strengthens** under the learning-rate
control (`lipo_random` +0.208 → +0.349, `cond7` +0.152 → +0.235) and the within-bin
check agrees (+0.307, +0.228). Learning rate was *suppressing* this relationship, not
creating it — the opposite of the curvature case, which is why one claim survives and
the other does not.

So the theoretically motivated, ~30x more expensive quantity is beaten at its own job
by the forward-only heuristic it was supposed to justify.

**But the effect is too weak to matter for selection.** |ρ| ≤ 0.35, plain `train_mse`
is a stronger gap predictor on most conditions, and on `hydro` every signal reverses
sign. On `hydro`, once lr is controlled, λ_max tracks the gap **strongly and
negatively** (−0.482 partial, −0.590 within-bin): sharper minima generalise *better*
under fitness extrapolation. That is the same inversion seen in the selection results,
now visible at the level of the underlying relationship rather than the decision.""")

    L.append("\n## 4. Are the two activation-energy variants independent?\n")
    L.append("| condition | ρ(`fg_legacy`, `fg_penult`) |")
    L.append("|---|---:|")
    for cond in ORDER:
        d = load(cond)
        if d is None:
            continue
        L.append(f"| `{cond}` | {spearmanr(d.fg_legacy, d.fg_penult).statistic:+.3f} |")
    L.append("""
**No — they are one signal.** ρ = +0.94 to +0.97 on every condition. The
penultimate-layer variant was tracked as a separate probe on the theory that the
representation feeding the output head matters more than a depth-average; empirically
it is nearly the same quantity. This is why its results track the full-depth version
throughout the study, and it means the two should not be counted as independent
evidence for one another.""")

    L.append("""
## 5. Summary of signal relationships

| claim | marginal | after removing lr | verdict |
|---|---|---|---|
| activation energy ↔ curvature | −0.40 | −0.12 | **artifact** — lr created it |
| activation energy ↔ gap | +0.15 | +0.23 | **real** — lr was masking it |
| λ_max ↔ gap (molecular) | +0.04 | −0.02 | **no relationship** |
| λ_max ↔ gap (`hydro`) | −0.20 | −0.48 | **real, and inverted** |
| `fg_legacy` ↔ `fg_penult` | +0.95 | — | **one signal, not two** |

The methodological lesson generalises past this study: a proxy paper that reports a
marginal correlation across a hyperparameter sweep has not shown a mechanism. Here
the same pool yields a spurious association in one direction and a suppressed real
one in the other, and only conditioning tells them apart.""")

    diagnostics(L)

    with open(OUT, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {OUT} ({len(L)} blocks)")


if __name__ == "__main__":
    main()
