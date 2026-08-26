#!/usr/bin/env python3
"""Per-dataset tables: every selector x every metric -> RESULTS_BY_DATASET.md.

One table per condition, for all four experiment families -- shared-pool selection
(phase1a), the fixed-architecture and validation-free ablations, and both sequential
families. Rows are selectors or search methods; columns are the three metrics the
relevant literature actually uses:

  MSE  (train-sigma standardised)  -- what this study optimises and selects on
  MAE  (raw target units)          -- what the TDC ADMET leaderboards report
  rho  (Spearman, raw)             -- what FLIP2 reports

The field trains on MSE (Chemprop's default, ridge regression) but reports MAE or
rank correlation, so a result that holds only under MSE is a statement about the
scoring choice rather than about selection. All three are therefore reported side by
side, each with a paired Wilcoxon p against that table's reference on that same
metric.

Selection rules are identical across the three metric columns -- only the metric used
to score the deployed model changes.
"""
from __future__ import annotations
import glob, os
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, rankdata

OUT = "artifacts/results/summaries/RESULTS_BY_DATASET.md"
P1A = "artifacts/results/phase1a_pool/candidates"
GD  = "artifacts/results/exploratory/gdsc/phase1a/candidates"
FS  = "artifacts/results/exploratory/featshift/phase1a/candidates"
SEV = "artifacts/results/exploratory/shift_severity/candidates"
ABL = "artifacts/results/ablations"

POOLS = [("caco2_random",P1A,""),("caco2_scaffold",P1A,""),("lipo_random",P1A,""),
         ("lipo_scaffold",P1A,""),("cond7",P1A,""),("amylase",P1A," ⚠ degenerate"),
         ("hydro",P1A,""),("gdsc_drug",GD," (post-hoc)"),
         ("featshift_high",FS," (post-hoc)"),("featshift_low",FS," (post-hoc)"),
         ("gdsc_l1000",FS," (post-hoc)")]
POOLS += [(f"shiftsev_{k}_a{a}",SEV," (synthetic)") for k in ("caco2","lipo")
          for a in ("000","025","050","075","100")]

# selectors for tables where a validation split exists
SELS = [("oracle","__or"),("val",None),("val+legacy","+fg_legacy"),
        ("val+penult","+fg_penult"),("val+hess","+hess_top"),
        ("fg_legacy","fg_legacy"),("fg_penult","fg_penult"),("hess_top","hess_top"),
        ("train_mse","__tr"),("n_params","__np"),("random(E)","__rand")]
# selectors for the val-free tables: no validation column exists at all (n_val = 0),
# so the reference becomes train MSE and the val+* rank-sums are undefined.
VF_SELS = [("oracle","__or"),("train_mse","__tr"),("train+legacy","t+fg_legacy"),
           ("train+penult","t+fg_penult"),("train+hess","t+hess_top"),
           ("fg_legacy","fg_legacy"),("fg_penult","fg_penult"),
           ("hess_top","hess_top"),("n_params","__np"),("random(E)","__rand")]
METRICS = [("test_mse_std","MSE",False),("test_mae_raw","MAE",False),
           ("test_spearman","rho",True)]

SEQ = ["random_val","tpe_val","hebo_val","tpe_fg_mo","hebo_fg_mo",
       "tpe_penult_mo","hebo_penult_mo","tpe_hess_mo","hebo_hess_mo"]
SIG = {"tpe_fg_mo":"fg_legacy","hebo_fg_mo":"fg_legacy","tpe_penult_mo":"fg_penult",
       "hebo_penult_mo":"fg_penult","tpe_hess_mo":"hess_top","hebo_hess_mo":"hess_top"}
PRE   = "artifacts/results/phase1b_sequential/runs"
EXTRA = "artifacts/results/exploratory/phase1b_extra/runs"
GDB   = "artifacts/results/exploratory/gdsc/phase1b/runs"
B1 = [("cond7",PRE,""),("hydro",PRE,""),("gdsc_drug",GDB," (post-hoc)"),
      ("caco2_random",EXTRA," (post-hoc)"),("caco2_scaffold",EXTRA," (post-hoc)"),
      ("lipo_random",EXTRA," (post-hoc)"),("lipo_scaffold",EXTRA," (post-hoc)"),
      ("amylase",EXTRA," (post-hoc) ⚠ degenerate")]

FOOT = ("*`Δ` is the **median paired difference** (negative = better than the "
        "reference), which the p-value refers to; the metric column is the median "
        "value, and the two can disagree when many runs tie. `b/w` = runs "
        "better/worse. **bold** = beats the reference on that metric at uncorrected "
        "p<0.05, direction taken from `b/w`. `undef.` = the metric is undefined for "
        "the selected model (Spearman of a constant predictor); such pairs are dropped "
        "from the test and counted in `b/w`. Holm-corrected views: "
        "`RESULTS_MASTER.md`.*")


def sel_idx(k, s):
    """Index of the row a selector picks. `+x` = rank-sum(val, x); `t+x` = rank-sum(train, x)."""
    if s is None: return k.val_mse_std.idxmin()
    if s == "__tr": return k.train_mse_std.idxmin()
    if s == "__np": return k.n_params.idxmin()
    if s == "__or": return k.test_mse_std.idxmin()
    if s.startswith("t+"):
        c=s[2:]; o=k[k[c].notna()]
        return o.index[int(np.argmin(rankdata(o.train_mse_std)+rankdata(o[c])))]
    if s.startswith("+"):
        c=s[1:]; o=k[k[c].notna()]
        return o.index[int(np.argmin(rankdata(o.val_mse_std)+rankdata(o[c])))]
    o=k[k[s].notna()]; return o[s].idxmin()


def runs(root, cond, need_val=True):
    out=[]
    for f in sorted(glob.glob(f"{root}/{cond}/*.parquet")):
        d=pd.read_parquet(f)
        d=d[d.status_train=="ok"]
        d=d[d.val_mse_std.notna()] if need_val else d[d.train_mse_std.notna()]
        d=d[(d.pred_std_test>=1e-6) & (d.dead_relu_frac<=0.5)]
        if {"hess_trace","weight_norm_fro"} <= set(d.columns):
            # Petzka et al. (2021) relative flatness, network-level approximation:
            # ||w||_F^2 * tr(H). Formed here rather than at train time because both
            # factors were already stored.
            d = d.assign(petzka=d.weight_norm_fro**2 * d.hess_trace)
        if len(d)>=5: out.append(d)
    return out


def cells_for(v, b, hib, is_ref):
    """One metric's four cells: median, median paired diff, better/worse, p."""
    med=float(np.median(v))
    # A NaN median is not a bug: Spearman is undefined for a constant predictor, so an
    # `undef.` cell is itself the degeneracy signature -- the deploy rule picked a dead
    # model. Surface it rather than hiding it behind a dropped row.
    if np.isfinite(med):
        f=np.asarray(v,dtype=float); f=f[np.isfinite(f)]
        q1,q3=np.percentile(f,[25,75])
        # both dispersions: IQR is what this study's paired analysis warrants, mean+/-sd
        # is what the TDC leaderboards print. sd uses ddof=1 (sample sd).
        sd=float(np.std(f,ddof=1)) if len(f)>1 else float("nan")
        mtxt=f"{med:.4f} [{q1:.4f}, {q3:.4f}] · {np.mean(f):.4f} ± {sd:.4f}"
    else:
        mtxt="undef."
    if is_ref: return [mtxt, "—", "—", "—"]
    d=(b-v) if hib else (v-b)
    nfin=int(np.sum(~np.isfinite(d)))
    d=d[np.isfinite(d)]
    p=float(wilcoxon(d).pvalue) if len(d)>2 and np.any(d!=0) else 1.0
    bt,ws=int((d<0).sum()),int((d>0).sum())
    # direction from the win/loss counts, NOT from median(d): rank-sum selectors tie
    # with the reference on many runs, so the median paired difference is often
    # exactly 0 while the test is decisive.
    star="**" if (p<0.05 and bt>ws) else ""
    if not np.isfinite(med): star=""            # never bold an undefined median
    sfx=f" ({nfin} undef.)" if nfin else ""
    return [f"{star}{mtxt}{star}", f"{np.median(d):+.4f}", f"{bt}/{ws}{sfx}", f"{p:.3f}"]


def pool_table(L, cond, root, note, sels, ref_name, ref_sp, need_val=True):
    R=runs(root,cond,need_val)
    if len(R)<5: return False
    pool=pd.concat(R)
    L.append(f"\n### `{cond}`{note}\n")
    L.append(f"{len(R)} outer runs · median {np.median([len(k) for k in R]):.0f}/48 "
             f"candidates kept · y_sigma = {pool.y_sigma.iloc[0]:.3f}\n")
    L.append("| selector | MSE: med [IQR] · mean ± sd | Δ vs ref | b/w | p | MAE: med [IQR] · mean ± sd | Δ | b/w | p | rho: med [IQR] · mean ± sd | Δ | b/w | p |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    base={m:np.array([R[i].loc[sel_idx(R[i],ref_sp),m] for i in range(len(R))])
          for m,_,_ in METRICS}
    for nm,sp in sels:
        cells=[]
        for m,_,hib in METRICS:
            v=(np.array([R[i][m].mean() for i in range(len(R))]) if sp=="__rand"
               else np.array([R[i].loc[sel_idx(R[i],sp),m] for i in range(len(R))]))
            cells += cells_for(v, base[m], hib, nm==ref_name)
        L.append(f"| `{nm}` | " + " | ".join(cells) + " |")
    L.append("")
    L.append(FOOT)
    return True


def seq_deploy(root, meth, cond, sig, rule):
    """Per outer run, the metrics of the trial the deploy rule picks from that
    method's own search trajectory. `rule` is the base objective: val or train."""
    base_col = "val_mse_std" if rule=="val" else "train_mse_std"
    out=[]
    for f in sorted(glob.glob(f"{root}/{meth}/{cond}/*.parquet")):
        g=pd.read_parquet(f)
        ok=g[(g.status_train=="ok") & g[base_col].notna()]
        if not len(ok): out.append(None); continue
        if sig is None:
            i=ok[base_col].idxmin()
        else:
            o=ok[ok[sig].notna()]
            if not len(o): out.append(None); continue
            i=o.index[int(np.argmin(rankdata(o[base_col])+rankdata(o[sig])))]
        out.append({m: float(ok.loc[i,m]) for m,_,_ in METRICS} if i in ok.index
                   else {m: float(o.loc[i,m]) for m,_,_ in METRICS})
    return out


def arr(rowsets, m):
    return np.array([np.nan if r is None else r[m] for r in rowsets])


def seq_table(L, cond, root, note, methods, ref, rule, oracle_from=None):
    got={mt: seq_deploy(root, mt, cond, SIG.get(mt), rule) for mt in methods}
    got={k:v for k,v in got.items() if v}
    if ref not in got: return False
    n=min(len(v) for v in got.values())
    L.append(f"\n### `{cond}`{note}\n")
    L.append(f"{n} outer runs x 48 trials · reference `{ref}`\n")
    L.append("| method | MSE: med [IQR] · mean ± sd | Δ vs ref | b/w | p | MAE: med [IQR] · mean ± sd | Δ | b/w | p | rho: med [IQR] · mean ± sd | Δ | b/w | p |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    base={m: arr(got[ref][:n], m) for m,_,_ in METRICS}
    for mt in methods:
        if mt not in got: continue
        cells=[]
        for m,_,hib in METRICS:
            cells += cells_for(arr(got[mt][:n], m), base[m], hib, mt==ref)
        L.append(f"| `{mt}` | " + " | ".join(cells) + " |")
    if oracle_from:
        # best test MSE actually visited by the reference arm's own search -- an upper
        # bound on what any deploy rule could have extracted from that trajectory.
        orc=[]
        for f in sorted(glob.glob(f"{root}/{oracle_from}/{cond}/*.parquet"))[:n]:
            g=pd.read_parquet(f); ok=g[g.status_train=="ok"]
            orc.append({m: float(ok.loc[ok.test_mse_std.idxmin(),m]) for m,_,_ in METRICS}
                       if len(ok) else None)
        cells=[]
        for m,_,hib in METRICS:
            cells += cells_for(arr(orc, m), base[m], hib, False)
        L.append(f"| `oracle (in `{oracle_from}` traj.)` | " + " | ".join(cells) + " |")
    L.append("")
    L.append(FOOT)
    return True


def main():
    L=["# RESULTS — per dataset, every selector, every metric",""]
    L.append("*Auto-generated by `scripts_icbinb/make_metric_tables.py`.*")
    L.append("")
    L.append(__doc__.split("\n\n",1)[1].strip())
    L.append("")
    L.append("Non-degenerate models only in the shared-pool sections (constant "
             "predictors and >50% dead-ReLU removed). Median over outer runs. `p` is a "
             "paired Wilcoxon against that table's reference **on that metric**; `—` "
             "marks the reference row itself. `random(E)` is the expected loss of "
             "choosing blind; `oracle` is unachievable and bounds what was available.")
    L.append("")
    L.append("**What an outer run is.** Not 10 random initialisations. An outer run is "
             "a complete independent replication of the whole pipeline, indexed by `r`, "
             "which fixes six seeds (`10000·r + 1 … +6`) governing the train/val/test "
             "split, **the 48-candidate Latin hypercube itself**, weight init, batch "
             "ordering, the stochastic proxy estimators, and the HPO sampler. So the "
             "candidate pool is redrawn every run — \"every selector ranks the same 48 "
             "models\" holds *within* a run, which is what makes these comparisons "
             "paired, not across runs. The test set is redrawn too on every condition "
             "**except** the two FLIP2 ones (`amylase`, `hydro`), whose split is a "
             "deterministic cut; their spread therefore reflects only training "
             "stochasticity and the pool redraw, and is not comparable to the molecular "
             "IQRs. Full definition: `RESULTS_MASTER.md` §2.1.")
    L.append("")
    L.append("**Dispersion — both conventions.** Each metric cell reads "
             "`median [25th, 75th] · mean ± sd` across outer runs (sample sd, ddof=1). "
             "Two are given because the field and this design want different things. "
             "The TDC leaderboards report **mean ± sd over 5 seeds**, so that form is "
             "here for comparability. But every contrast in this study is **paired "
             "within an outer run** — all selectors rank the same 48 models, from the "
             "same split and initialisation — and a standard deviation invites the "
             "unpaired comparison that throws that pairing away; on 10–30 runs of a "
             "non-normal, heavily tied distribution the paired Wilcoxon signed-rank is "
             "both more powerful and less assumption-laden. So **the `p` columns carry "
             "the inference and both dispersions are descriptive only.** Two "
             "consequences to read carefully: where two selectors' intervals overlap "
             "almost entirely and `p` is still small, that is the pairing doing the "
             "work, not an error; and where the mean sits well outside the IQR the "
             "distribution is skewed by a few bad runs, so the median is the more "
             "faithful summary of a typical deployment.")
    L.append("")
    L.append("| section | family | reference |")
    L.append("|---|---|---|")
    L.append("| §0 | **what every method is** — signals, selectors, search methods | — |")
    L.append("| §1 | `phase1a` shared pool | `val` |")
    L.append("| §2 | `fixed_arch` ablation (256x2 pinned) | `val` |")
    L.append("| §3 | `valfree_pool` ablation (n_val = 0) | `train_mse` |")
    L.append("| §4 | `phase1b` sequential search | `tpe_val` |")
    L.append("| §5 | `valfree_sequential` ablation | per-arm `train_mse` |")
    L.append("| §6 | `trace` run — Hessian trace + relative flatness | `val` |")

    n1=n2=n3=n4=n5=0
    L.append("\n## 0. What every method in these tables is\n")
    L.append("Notation: a pool has 48 trained candidates; $A_\\ell$ is the "
             "**post-activation** output of hidden layer $\\ell$ (after BatchNorm → "
             "ReLU → dropout) on a fixed batch of $B$ training examples at width $w$; "
             "$H$ is the Hessian of the **training** MSE w.r.t. all parameters; $w$ "
             "also denotes the flattened weight vector where unambiguous. Every signal "
             "is computed from **training data only** — that is the whole point; a "
             "signal needing held-out data is not a replacement for held-out data.\n")

    L.append("### 0.1 The five signals\n")
    L.append("| signal | one line | formula / algorithm |")
    L.append("|---|---|---|")
    L.append("| `fg_legacy` | per-neuron **activation energy**, averaged over all "
             "hidden layers | $\\max_b \\operatorname{mean}_\\ell \\lVert A_\\ell "
             "\\rVert_F^2 / (B w)$ — mean over layers, max over fixed proxy batches |")
    L.append("| `fg_penult` | the same energy read off the **last hidden layer** only "
             "— the representation the output head sees | $\\max_b \\lVert A_L "
             "\\rVert_F^2 / (B w)$ |")
    L.append("| `hess_top` | **top Hessian eigenvalue** $\\lambda_{\\max}$ — true "
             "curvature in the single sharpest direction | power iteration on exact "
             "Hessian-vector products (Pearlmutter double-backward), ≤30 iterations, "
             "relative tol $10^{-3}$ |")
    L.append("| `hess_trace` | **Hessian trace** $\\operatorname{tr}(H)$ — bulk "
             "curvature, the sum of *all* eigenvalues | Hutchinson estimator "
             "$\\operatorname{tr}(H)=\\mathbb{E}_z[z^\\top H z]$ with $z$ Rademacher, "
             "**30 probes**, one HVP each |")
    L.append("| `petzka` | **relative flatness** (Petzka et al., NeurIPS 2021), "
             "network-level approximation — a *norm-weighted* trace | $\\lVert w "
             "\\rVert_F^2 \\cdot \\operatorname{tr}(H)$. ⚠ the paper's measure is a "
             "layer-wise quadratic form $\\sum_{ij} w_i w_j \\partial^2 "
             "L/\\partial w_i \\partial w_j$; this is a surrogate formed from stored "
             "quantities |")
    L.append("\n> ⚠ **Only `hess_top`, `hess_trace` and `petzka` measure curvature.** "
             "`fg_legacy` and `fg_penult` measure activation energy, which was merely "
             "*motivated* as a cheap sharpness stand-in — and this study shows that "
             "motivation is wrong: conditioned on learning rate they carry almost no "
             "$\\lambda_{\\max}$ information and are an *inverse* measure of "
             "$\\operatorname{tr}(H)$. \"Train-only geometry signals\" is the accurate "
             "collective term. See `RESULTS_MASTER.md` §13 and §13.1.\n")

    L.append("### 0.2 Selectors (§1–§3, §6)\n")
    L.append("A **selector** picks one model from the pool using training and "
             "validation information only, and we then report that model's test score. "
             "**`A+B` means rank-sum**: rank all candidates by A, rank by B, add the "
             "ranks, take the minimum — $\\arg\\min_i [\\operatorname{rank}(a_i) + "
             "\\operatorname{rank}(b_i)]$. Rank-sum is scale-free; a product is "
             "dominated by whichever term has the larger dynamic range, and a weighted "
             "sum needs a weight we would have had to tune, quietly reintroducing the "
             "validation set we are trying to eliminate.\n")
    L.append("| selector | picks the candidate with… | why it is here |")
    L.append("|---|---|---|")
    for nm, desc, why in [
        ("`val`", "lowest validation MSE", "what a practitioner actually does — the reference in §1, §2, §6"),
        ("`train_mse`", "lowest training MSE", "the val-free alternative, and a strong baseline; the reference in §3"),
        ("`fg_legacy` / `fg_penult` / `hess_top` / `hess_trace` / `petzka`",
         "lowest value of that signal alone", "pure train-only selection — no validation at all"),
        ("`val+legacy` / `val+penult` / `val+hess` / `val+trace` / `val+petzka`",
         "lowest rank(val MSE) + rank(signal)", "*augment* validation rather than replace it"),
        ("`train+legacy` / `train+penult` / `train+hess`",
         "lowest rank(train MSE) + rank(signal)", "the §3 val-free ablations, where no validation set exists"),
        ("`n_params`", "fewest parameters", "capacity heuristic with no geometry in it — a deliberately dumb baseline"),
        ("`random(E)`", "*(not a selector)* the **mean** test loss over the pool",
         "E[loss] under a uniform random pick — the cost of choosing blind. A selector that cannot clear this carries no usable signal"),
        ("`oracle`", "lowest **test** MSE", "not achievable — bounds what was on the table"),
    ]:
        L.append(f"| {nm} | {desc} | {why} |")
    L.append("\n`oracle` and `random(E)` are the only rows that touch the test set; "
             "every real selector is blind to it.\n")

    L.append("### 0.3 Sequential search methods (§4–§5)\n")
    L.append("These are **not** selectors over a shared pool — each method runs its "
             "own 48-trial hyperparameter search, so the rows compare *search "
             "strategies*. Names are `{sampler}_{objective}`.\n")
    L.append("| component | value | what it is |")
    L.append("|---|---|---|")
    L.append("| sampler | `random` | uniform sampling over the search space |")
    L.append("| | `tpe` | Optuna `TPESampler(multivariate=True)`; **MOTPE** "
             "(non-dominated sorting, hypervolume tie-break) for the multi-objective arms |")
    L.append("| | `hebo` | HEBO; multi-objective needs "
             "`hebo.optimizers.general.GeneralBO(num_obj=2)` — vector LCB + NSGA-II, "
             "$\\kappa = 2.0$, 8 random initial points — because the plain `HEBO` class "
             "has no `num_obj` |")
    L.append("| objective | `_val` | **single**-objective on validation MSE |")
    L.append("| | `_fg_mo` / `_penult_mo` / `_hess_mo` | **multi**-objective on (base "
             "MSE, signal), where the signal is `fg_legacy` / `fg_penult` / "
             "`hess_top`. Neither sampler scalarises, so no weight had to be chosen |")
    L.append("\nAll nine arms, spelled out — the §4 base objective is validation MSE, "
             "the §5 base objective is training MSE:\n")
    L.append("| method | sampler | optimises |")
    L.append("|---|---|---|")
    for nm, samp, obj in [
        ("`random_val`", "uniform random", "base MSE only"),
        ("`tpe_val`", "Optuna TPE", "base MSE only — **the reference row in §4**"),
        ("`hebo_val`", "HEBO", "base MSE only"),
        ("`tpe_fg_mo`", "Optuna MOTPE", "(base MSE, `fg_legacy`) jointly"),
        ("`hebo_fg_mo`", "HEBO `GeneralBO(num_obj=2)`", "(base MSE, `fg_legacy`) jointly"),
        ("`tpe_penult_mo`", "Optuna MOTPE", "(base MSE, `fg_penult`) jointly"),
        ("`hebo_penult_mo`", "HEBO `GeneralBO(num_obj=2)`", "(base MSE, `fg_penult`) jointly"),
        ("`tpe_hess_mo`", "Optuna MOTPE", "(base MSE, `hess_top`) jointly"),
        ("`hebo_hess_mo`", "HEBO `GeneralBO(num_obj=2)`", "(base MSE, `hess_top`) jointly"),
    ]:
        L.append(f"| {nm} | {samp} | {obj} |")
    L.append("\nTwo further row labels appear in these sections. "
             "`oracle (in X traj.)` is the lowest **test** MSE among the trials method "
             "X actually visited — an upper bound on what any deploy rule could have "
             "extracted from that trajectory, not something achievable. `train_mse "
             "(ref, on X traj.)` is the §5 baseline: that same trajectory deployed by "
             "training MSE alone, which is what each §5 row is compared against.\n")
    L.append("\n**Deploy rule** (how a finished search yields one model): `*_val` arms "
             "take the trial with lowest base MSE; `*_mo` arms take the within-budget "
             "rank-sum of base MSE and the signal over all observed trials.\n")
    L.append("> ⚠ **The same six `_mo` names mean different methods in §4 and §5.** In "
             "§4 the base objective is **validation** MSE. In §5 it is **training** MSE "
             "($n_{val} = 0$ — validation is folded into training and no validation set "
             "exists). So `tpe_fg_mo` optimises (val MSE, `fg_legacy`) in §4 and (train "
             "MSE, `fg_legacy`) in §5. Always read the method name together with its "
             "section.\n")

    L.append("\n## 1. `phase1a` — shared-pool selection\n")
    L.append("Every selector ranks the same 48 trained models, so comparisons are "
             "paired and no selector gets a different search.")
    for cond, root, note in POOLS:
        n1 += pool_table(L, cond, root, note, SELS, "val", None)

    L.append("\n## 2. `fixed_arch` — architecture pinned to 256x2\n")
    L.append("Only learning rate, weight decay and dropout vary. Tests whether the "
             "failure is a cross-architecture comparability problem. Note `n_params` is "
             "constant here, so that row degenerates to an arbitrary pick and is "
             "reported only for completeness.")
    for cond in ["cond7","hydro"]:
        n2 += pool_table(L, cond, f"{ABL}/fixed_arch", "", SELS, "val", None)

    L.append("\n## 3. `valfree_pool` — validation folded into training\n")
    L.append("`n_val = 0`: the validation split is added to training, so **no "
             "validation column exists** and the `val+*` rank-sums are undefined. The "
             "reference is therefore `train_mse`, and this table answers a different "
             "question from §1 — *given that you have no validation set, what should "
             "you select on?* The cross-tree comparison against the matching `phase1a` "
             "`val` cell (same seeds, identical test split) is in `RESULTS_MASTER.md` §11.")
    for cond in ["caco2_scaffold","lipo_scaffold","amylase","hydro"]:
        note=" ⚠ degenerate" if cond=="amylase" else ""
        n3 += pool_table(L, cond, f"{ABL}/valfree_pool", note, VF_SELS,
                         "train_mse", "__tr", need_val=False)

    L.append("\n## 4. `phase1b` — sequential search, can a signal *drive* it?\n")
    L.append("Each method runs its **own** 48-trial search, so these rows compare "
             "search strategies rather than selectors over a shared pool. Deployment "
             "picks the trial minimising validation MSE (`*_val` arms) or the rank-sum "
             "of validation MSE and the signal (`*_mo` arms), from that method's own "
             "trajectory. Following the pre-registered Phase 1B analysis, **the "
             "degeneracy filter is not applied here** — a search trajectory is not a "
             "fixed pool, and filtering it would change what each method is credited "
             "with having found. This is why `amylase` is flagged: its trajectories "
             "contain the constant predictors that §1 removes.")
    for cond, root, note in B1:
        n4 += seq_table(L, cond, root, note, SEQ, "tpe_val", "val", oracle_from="tpe_val")

    L.append("\n## 5. `valfree_sequential` — search with no validation at all\n")
    L.append("The six multi-objective arms searching **`(train_mse, signal)`** instead "
             "of `(val_mse, signal)`. Deployment uses the train-side rank-sum. Here the "
             "reference is **per-arm**: each row is compared against *that same "
             "trajectory* deployed by `train_mse` alone, which is the contrast that "
             "isolates what the signal adds once validation is gone. The `train_mse` row "
             "shows the reference deployed on `tpe_fg_mo`'s trajectory for scale.")
    for cond in ["cond7","hydro"]:
        root=f"{ABL}/valfree_sequential"
        meths=[m for m in SEQ if m in SIG]
        got={mt: seq_deploy(root, mt, cond, SIG[mt], "train") for mt in meths}
        bas={mt: seq_deploy(root, mt, cond, None, "train") for mt in meths}
        got={k:v for k,v in got.items() if v}
        if not got: continue
        n=min(len(v) for v in got.values())
        L.append(f"\n### `{cond}`\n")
        L.append(f"{n} outer runs x 48 trials · reference = same trajectory deployed by "
                 f"`train_mse` alone\n")
        L.append("| method | MSE [IQR] | Δ vs ref | b/w | p | MAE [IQR] | Δ | b/w | p | rho [IQR] | Δ | b/w | p |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for mt in meths:
            if mt not in got: continue
            cells=[]
            for m,_,hib in METRICS:
                cells += cells_for(arr(got[mt][:n],m), arr(bas[mt][:n],m), hib, False)
            L.append(f"| `{mt}` | " + " | ".join(cells) + " |")
        ref0=meths[0]
        cells=[]
        for m,_,hib in METRICS:
            cells += cells_for(arr(bas[ref0][:n],m), arr(bas[ref0][:n],m), hib, True)
        L.append(f"| `train_mse` (ref, on `{ref0}` traj.) | " + " | ".join(cells) + " |")
        L.append("")
        L.append(FOOT)
        n5 += 1

    L.append("\n## 6. `trace` — does a better curvature summary select better?\n")
    L.append("A separate 3-condition run measuring the Hessian **trace** (Hutchinson, "
             "30 Rademacher probes) alongside λ_max, plus `petzka` = `‖w‖²·tr(H)`, an "
             "approximation to Petzka et al.'s relative flatness formed from stored "
             "quantities. λ_max is a median 0.12–0.14 of the trace, so `hess_top` sees "
             "roughly an eighth of the total curvature; these tables ask whether "
             "measuring the rest helps. **5 outer runs per condition** — sized to "
             "estimate correlations, not selection wins, so read Δ and `b/w` rather "
             "than `p`. Mechanism and the norm-vs-curvature decomposition: "
             "`RESULTS_trace.md`; summary in `RESULTS_MASTER.md` §13.1.")
    TRACE_SELS = [("oracle","__or"),("val",None),("val+legacy","+fg_legacy"),
                  ("val+hess","+hess_top"),("val+trace","+hess_trace"),
                  ("val+petzka","+petzka"),("fg_legacy","fg_legacy"),
                  ("hess_top","hess_top"),("hess_trace","hess_trace"),
                  ("petzka (approx.)","petzka"),("train_mse","__tr"),
                  ("n_params","__np"),("random(E)","__rand")]
    n6=0
    for cond in ["caco2_scaffold","cond7","hydro"]:
        n6 += pool_table(L, cond, "artifacts/results/exploratory/trace/candidates",
                         "", TRACE_SELS, "val", None)

    L.append(f"\n---\n\n**Coverage.** §6 `trace` {n6} conditions · §1 `phase1a` {n1} conditions · §2 `fixed_arch` "
             f"{n2} · §3 `valfree_pool` {n3} · §4 `phase1b` {n4} · §5 "
             f"`valfree_sequential` {n5} — {n1+n2+n3+n4+n5+n6} tables.")
    os.makedirs(os.path.dirname(OUT),exist_ok=True)
    open(OUT,"w").write("\n".join(L)+"\n")
    print(f"wrote {OUT} ({n1+n2+n3+n4+n5+n6} tables: "
          f"1a={n1} fixed={n2} vfpool={n3} 1b={n4} vfseq={n5} trace={n6})")

if __name__=="__main__":
    main()
