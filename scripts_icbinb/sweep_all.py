#!/usr/bin/env python3
"""One master sweep over EVERY completed experiment -> RESULTS_MASTER.md.

Single decision rule everywhere so the cells are comparable:
  * degenerate models removed (constant predictors, >50% dead ReLU)
  * paired within outer run
  * two references: `val` (what a practitioner uses) and `random(E)` (the bar any
    signal must clear to be worth anything)
  * Holm within each condition's selector family
"""
from __future__ import annotations
import glob, os
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, rankdata, spearmanr

OUT = "artifacts/results/summaries/RESULTS_MASTER.md"
P1A = "artifacts/results/phase1a_pool/candidates"
GD  = "artifacts/results/exploratory/gdsc/phase1a/candidates"
SEV = "artifacts/results/exploratory/shift_severity/candidates"
FS  = "artifacts/results/exploratory/featshift/phase1a/candidates"
FIX = "artifacts/results/ablations/fixed_arch"
# (heading, [(condition, root, note)])
# gdsc_drug runs the IDENTICAL Phase-1A protocol -- same 48-candidate pool, same
# selectors, same analysis -- so it belongs in the main table. It carries a note
# because it was added after the protocol was unblinded and therefore cannot join
# the pre-registered CONFIRMATORY family; that is a statistical bookkeeping matter,
# not a reason to treat the measurement as second class.
POOLS = [
    ("phase1a", "Shared-pool selection — every selector ranks the same 48 models", [
        ("caco2_random",   P1A, ""), ("caco2_scaffold", P1A, ""),
        ("lipo_random",    P1A, ""), ("lipo_scaffold",  P1A, ""),
        ("cond7",          P1A, ""), ("amylase",        P1A, ""),
        ("hydro",          P1A, ""),
        ("gdsc_drug",      GD,  "added after unblinding"),
        ("featshift_high", FS,  "added after unblinding"),
        ("featshift_low",  FS,  "added after unblinding"),
    ]),
    ("fixed_arch", "Architecture pinned to 256x2; only lr / weight-decay / dropout vary",
     [("cond7",FIX,""),("hydro",FIX,"")]),
    ("shiftsev", "Synthetic severity knob: alpha = 0 is IID, alpha = 1 is pure feature extrapolation",
     [(f"shiftsev_{k}_a{a}", SEV, "") for k in ("caco2","lipo") for a in ("000","025","050","075","100")]),
]
SEL = [("val",None),("fg_legacy","fg_legacy"),("fg_penult","fg_penult"),("hess_top","hess_top"),
       ("val+legacy","+fg_legacy"),("val+penult","+fg_penult"),("val+hess","+hess_top"),
       ("train_mse","__tr"),("n_params","__np")]

def holm(ps):
    n=len(ps); a=np.empty(n); r=0.0
    for i,k in enumerate(np.argsort(ps)): r=max(r,(n-i)*ps[k]); a[k]=min(r,1.0)
    return a

def pick(k,s):
    if s=="__rand": return float(k.test_mse_std.mean())
    if s=="__or":   return float(k.test_mse_std.min())
    if s=="__tr":   return float(k.loc[k.train_mse_std.idxmin(),"test_mse_std"])
    if s=="__np":   return float(k.loc[k.n_params.idxmin(),"test_mse_std"])
    if s is None:   return float(k.loc[k.val_mse_std.idxmin(),"test_mse_std"])
    if s.startswith("+"):
        c=s[1:]; o=k[k[c].notna()]
        if not len(o): return np.nan
        return float(o.iloc[int(np.argmin(rankdata(o.val_mse_std)+rankdata(o[c])))].test_mse_std)
    o=k[k[s].notna()]
    return float(o.loc[o[s].idxmin(),"test_mse_std"]) if len(o) else np.nan

def runs(root,cond):
    out=[]
    for f in sorted(glob.glob(f"{root}/{cond}/*.parquet")):
        d=pd.read_parquet(f)
        d=d[(d.status_train=="ok") & d.val_mse_std.notna()]
        d=d[(d.pred_std_test>=1e-6) & (d.dead_relu_frac<=0.5)]
        if len(d)>=5: out.append(d)
    return out


def iqr(v):
    """`median [25th, 75th] . mean +/- sd` across outer runs (sample sd, ddof=1).

    Both dispersions are given: mean +/- sd is what the TDC leaderboards print, so it
    is here for comparability; the IQR is what this design warrants, since every
    contrast is paired within an outer run and a std invites the unpaired comparison
    that discards the pairing. Descriptive only -- the `p` columns carry the inference.
    """
    v=np.asarray(v,dtype=float); v=v[np.isfinite(v)]
    if not len(v): return "—"
    q1,q3=np.percentile(v,[25,75])
    sd=float(np.std(v,ddof=1)) if len(v)>1 else float("nan")
    return f"{np.median(v):.4f} [{q1:.4f}, {q3:.4f}] · {np.mean(v):.4f} ± {sd:.4f}"


def phase1b(L):
    """Sequential HPO summary. Structurally different from the shared-pool tables:
    here each METHOD runs its own 48-trial search, so we compare methods, not
    selectors over one pool. Full five-panel detail is in RESULTS_phase1b.md."""
    # Phase 1B lives in THREE trees: the pre-registered pair, the post-hoc gdsc arm,
    # and the post-hoc arm on the five conditions the plan excluded from 1B (§10.6).
    # Reading only the first silently dropped 6 of the 8 conditions actually run.
    PRE = "artifacts/results/phase1b_sequential/runs"
    EXTRA = "artifacts/results/exploratory/phase1b_extra/runs"
    GDB = "artifacts/results/exploratory/gdsc/phase1b/runs"
    TREES = [("cond7", PRE, ""), ("hydro", PRE, ""), ("gdsc_drug", GDB, "post-hoc"),
             ("caco2_random", EXTRA, "post-hoc"), ("caco2_scaffold", EXTRA, "post-hoc"),
             ("lipo_random", EXTRA, "post-hoc"), ("lipo_scaffold", EXTRA, "post-hoc"),
             ("amylase", EXTRA, "post-hoc, degenerate")]
    R = PRE
    if not os.path.isdir(R):
        return
    METH = ["random_val","tpe_val","hebo_val","tpe_fg_mo","hebo_fg_mo",
            "tpe_penult_mo","hebo_penult_mo","tpe_hess_mo","hebo_hess_mo"]
    NAT = {"tpe_fg_mo":"fg_legacy","hebo_fg_mo":"fg_legacy","tpe_penult_mo":"fg_penult",
           "hebo_penult_mo":"fg_penult","tpe_hess_mo":"hess_top","hebo_hess_mo":"hess_top"}
    def dep(g, sig):
        ok = g[(g.status_train=="ok") & g.val_mse_std.notna()]
        if not len(ok): return np.nan
        if sig is None: return float(ok.loc[ok.val_mse_std.idxmin(),"test_mse_std"])
        o = ok[ok[sig].notna()]
        if not len(o): return np.nan
        return float(o.iloc[int(np.argmin(rankdata(o.val_mse_std)+rankdata(o[sig])))].test_mse_std)
    L.append("\n## 10. `phase1b` — can a signal *drive* the search?\n")
    L.append("Phase 1B. Each method runs its **own** 48-trial search, so this compares "
             "search strategies, not selectors over a shared pool. Reference is "
             "`tpe_val` (TPE on validation MSE). Holm over the 8 contrasts per "
             "condition. Full detail: `RESULTS_phase1b.md`.\n")
    L.append("\n### All eight conditions — which methods Holm-beat `tpe_val`?\n")
    tally = []
    L.append("| condition | note | reps | better than `tpe_val` | worse than `tpe_val` |")
    L.append("|---|---|---:|---|---|")
    for cond, root, note in TREES:
        fl = {m: sorted(glob.glob(f"{root}/{m}/{cond}/*.parquet")) for m in METH}
        if not fl.get("tpe_val"):
            continue
        VV = {m: np.array([dep(pd.read_parquet(f), NAT.get(m)) for f in fs])
              for m, fs in fl.items() if fs}
        bb = VV["tpe_val"]
        nm, pp, mm = [], [], []
        for m in METH:
            if m == "tpe_val" or m not in VV:
                continue
            k = min(len(VV[m]), len(bb))
            d = VV[m][:k] - bb[:k]
            d = d[np.isfinite(d)]
            if not len(d):
                continue
            pp.append(float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0)
            nm.append(m)
            mm.append(float(np.median(d)))
        if not pp:
            continue
        aa = holm(np.array(pp))
        bet = [n for n, md, a in zip(nm, mm, aa) if a < 0.05 and md < 0]
        wor = [n for n, md, a in zip(nm, mm, aa) if a < 0.05 and md > 0]
        L.append(f"| `{cond}` | {note} | {len(bb)} | "
                 f"{', '.join('`' + x + '`' for x in bet) or '—'} | "
                 f"{', '.join('`' + x + '`' for x in wor) or '—'} |")
        tally.append((cond, note, bet, wor))
    nb = [c for c, nt, b, w in tally if b and "degenerate" not in nt]
    nw = [c for c, nt, b, w in tally if w and "degenerate" not in nt]
    L.append(f"\nAcross the eight conditions, proxy-steered search Holm-beats "
             f"validation-driven search on **{len(nb)}** non-degenerate condition(s) "
             f"({', '.join('`'+c+'`' for c in nb) or 'none'}) and is Holm-**worse** on "
             f"**{len(nw)}** ({', '.join('`'+c+'`' for c in nw) or 'none'}).\n")
    L.append("The one clear win is `lipo_scaffold`, where all four activation-energy "
             "arms improve on `tpe_val` by 2-5% (9/1 runs each) while both curvature "
             "arms do nothing — and where the *random* split shows no effect at all, so "
             "it is the scaffold shift that the signal is responding to. Set against "
             "that, the same family is Holm-worse on `gdsc_drug`, and curvature-steered "
             "search is Holm-worse on `cond7`. The direction is condition-dependent, "
             "which is the same instability the shared-pool tables show.\n")
    L.append("\nThe two pre-registered conditions in full:\n")
    for cond in ["cond7","hydro"]:
        files = {m: sorted(glob.glob(f"{R}/{m}/{cond}/*.parquet")) for m in METH}
        if not files["tpe_val"]:
            continue
        V = {m: np.array([dep(pd.read_parquet(f), NAT.get(m)) for f in fs])
             for m, fs in files.items() if fs}
        base = V["tpe_val"]
        rows, ps = [], []
        for m in METH:
            if m == "tpe_val" or m not in V: continue
            d = V[m]-base; d = d[np.isfinite(d)]
            ps.append(float(wilcoxon(d).pvalue) if np.any(d!=0) else 1.0)
            rows.append((m, float(np.median(V[m])), float(np.median(d)),
                         int((d<0).sum()), int((d>0).sum())))
        adj = holm(np.array(ps))
        L.append(f"\n### `{cond}` — {len(base)} outer runs x 48 trials\n")
        L.append("| method | deployed test: med [IQR] · mean ± sd | Δ vs `tpe_val` | better/worse | p | p_holm |")
        L.append("|---|---:|---:|---:|---:|---:|")
        L.append(f"| `tpe_val` (reference) | {iqr(base)} | — | — | — | — |")
        for (m, mv, md, b, w), pr, a in zip(rows, ps, adj):
            mark = " **" if a < 0.05 else ""
            L.append(f"| `{m}` | {iqr(V[m])} | {md:+.4f} | {b}/{w} | {pr:.3f} | {a:.3f}{mark} |")
    L.append("""
**Reading this.** No proxy-steered method Holm-beats validation-driven search on
either condition. On `hydro` three arms look significantly *worse* by raw p
(0.014-0.037) but none survive Holm (0.109-0.223) — which is exactly the case worked
through in section 4: with 8 contrasts, three hits at that level is close to what
noise produces, so the honest statement is *not established*, not *shown to hurt*.

The informative result in Phase 1B is not the method ranking but the **deploy-rule
decomposition** (`RESULTS_phase1b.md` panel c): every stored trial carries val, train
and all three signals, so any deploy rule can be applied post hoc to any search
trajectory. Pooled over all nine searches (n=90 paired), the augmentation that helps
on `cond7` **hurts** on `hydro`, and vice versa for curvature — the sign flip that
finding (c) rests on.""")



def valfree(L):
    """The two val-free ablations: is validation data better spent as training data?"""
    VP = "artifacts/results/ablations/valfree_pool"
    VS = "artifacts/results/ablations/valfree_sequential"
    def sel_vf(k, s):
        if s == "__rand": return float(k.test_mse_std.mean())
        if s == "__or":   return float(k.test_mse_std.min())
        if s == "__tr":   return float(k.loc[k.train_mse_std.idxmin(),"test_mse_std"])
        c = s[2:] if s.startswith("t+") else s
        o = k[k[c].notna()]
        if not len(o): return np.nan
        if s.startswith("t+"):
            return float(o.iloc[int(np.argmin(rankdata(o.train_mse_std)+rankdata(o[c])))].test_mse_std)
        return float(o.loc[o[c].idxmin(),"test_mse_std"])
    def runs_vf(root, cond):
        out=[]
        for f in sorted(glob.glob(f"{root}/{cond}/*.parquet")):
            d=pd.read_parquet(f); d=d[d.status_train=="ok"]
            d=d[(d.pred_std_test>=1e-6)&(d.dead_relu_frac<=0.5)]
            if len(d)>=5: out.append(d)
        return out

    if os.path.isdir(VP):
        L.append("\n## 11. `valfree_pool` — validation folded into training\n")
        L.append("`n_val = 0`: the validation split is added to training, so no validation "
                 "signal exists and only train-side selectors are defined. The practitioner's "
                 "question — **is validation data better spent as training data?** The `val "
                 "(main)` column is the matching `phase1a` cell, paired by outer run: same "
                 "candidates, same seeds, **identical test split**.\n")
        L.append("| condition | reps | val (main) | train_mse | Δ | better/worse | p | best val-free selector |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---|")
        for cond in ["caco2_scaffold","lipo_scaffold","amylase","hydro"]:
            V=runs_vf(VP,cond)
            if len(V)<3: continue
            M={os.path.basename(f):pd.read_parquet(f) for f in glob.glob(f"{P1A}/{cond}/*.parquet")}
            Vd={os.path.basename(f):pd.read_parquet(f) for f in glob.glob(f"{VP}/{cond}/*.parquet")}
            ks=sorted(set(M)&set(Vd))
            if len(ks)<3: continue
            def clean(d):
                d=d[d.status_train=="ok"]; return d[(d.pred_std_test>=1e-6)&(d.dead_relu_frac<=0.5)]
            a=np.array([float(clean(M[k]).loc[clean(M[k]).val_mse_std.idxmin(),"test_mse_std"]) for k in ks])
            b=np.array([sel_vf(clean(Vd[k]),"__tr") for k in ks])
            d_=b-a; pv=float(wilcoxon(d_).pvalue) if np.any(d_!=0) else 1.0
            cand={n:np.array([sel_vf(clean(Vd[k]),sp) for k in ks]) for n,sp in
                  [("train_mse","__tr"),("train+legacy","t+fg_legacy"),("train+penult","t+fg_penult"),
                   ("train+hess","t+hess_top"),("fg_legacy","fg_legacy"),("fg_penult","fg_penult")]}
            best=min(cand,key=lambda n:np.median(cand[n]))
            flag=" ⚠" if cond=="amylase" else ""
            L.append(f"| `{cond}`{flag} | {len(ks)} | {iqr(a)} | {iqr(b)} | "
                     f"{np.median(d_):+.4f} | {int((d_<0).sum())}/{int((d_>0).sum())} | {pv:.3f} | "
                     f"`{best}` {np.median(cand[best]):.4f} |")
        L.append("\nPositive Δ means giving up the validation set costs you. **At parity on every "
                 "non-degenerate condition** — you can discard the validation split, train on it, "
                 "select on training MSE, and lose nothing measurable. `amylase` ⚠ is the "
                 "degenerate condition and is not evidence.")

    if os.path.isdir(VS):
        L.append("\n## 12. `valfree_seq` — sequential search with no validation at all\n")
        L.append("The six multi-objective arms searching **`(train_mse, signal)`** instead of "
                 "`(val_mse, signal)` — see the name-collision warning in §5.4. Reference is "
                 "`train_mse` alone, so this asks whether the signal adds anything once "
                 "validation is gone entirely.\n")
        SIG={"tpe_fg_mo":"fg_legacy","hebo_fg_mo":"fg_legacy","tpe_penult_mo":"fg_penult",
             "hebo_penult_mo":"fg_penult","tpe_hess_mo":"hess_top","hebo_hess_mo":"hess_top"}
        for cond in ["cond7","hydro"]:
            rows,ps=[],[]
            for m,sg in SIG.items():
                fs=sorted(glob.glob(f"{VS}/{m}/{cond}/*.parquet"))
                if not fs: continue
                A,B=[],[]
                for f in fs:
                    g=pd.read_parquet(f); g=g[g.status_train=="ok"]
                    o=g[g[sg].notna()]
                    A.append(float(o.iloc[int(np.argmin(rankdata(o.train_mse_std)+rankdata(o[sg])))].test_mse_std))
                    B.append(float(g.loc[g.train_mse_std.idxmin(),"test_mse_std"]))
                A,B=np.array(A),np.array(B); d_=A-B
                ps.append(float(wilcoxon(d_).pvalue) if np.any(d_!=0) else 1.0)
                rows.append((m,float(np.median(A)),float(np.median(B)),float(np.median(d_)),
                             int((d_<0).sum()),int((d_>0).sum())))
            if not rows: continue
            adj=holm(np.array(ps))
            L.append(f"\n### `{cond}` — 10 outer runs x 48 trials\n")
            L.append("| method | train+signal | train_mse | Δ | better/worse | p | p_holm |")
            L.append("|---|---:|---:|---:|---:|---:|---:|")
            for (m,a,b,md,w,n),pr,ah in zip(rows,ps,adj):
                mk=" **" if ah<0.05 and md<0 else (" ✗" if ah<0.05 else "")
                L.append(f"| `{m}` | {a:.4f} | {b:.4f} | {md:+.4f} | {w}/{n} | {pr:.3f} | {ah:.3f}{mk} |")
        L.append("""
**The sign flip survives with validation deleted.** On `cond7` **all six** arms
Holm-beat train-MSE alone. On `hydro` none do and one is significantly **worse**.
Same signals, same code, opposite conclusion — decided only by which shift is faced.
This is the strongest form of the finding, because here the signal is not a
tie-breaker on top of validation; it is doing the work.""")


def signals_and_degeneracy(L):
    """Why it fails, and the artifact that nearly hid it."""
    L.append("""
## 13. Why it fails — signal relationships

Full tables and method in `RESULTS_signals.md`. A correlation taken across a
hyperparameter sweep is not evidence of a mechanism: a shared driver can manufacture
an association or mask one, and **both happen here, in opposite directions.** Each
relationship is therefore reported marginally, as a rank-residualised partial given
architecture and learning rate, and again within narrow lr bins.

| relationship | marginal ρ | after removing lr | verdict |
|---|---:|---:|---|
| `fg_legacy` ↔ λ_max | −0.40 | **−0.12** | **artifact** — lr created it |
| `fg_legacy` ↔ generalisation gap | +0.15 | **+0.23** | **real** — lr was masking it |
| λ_max ↔ gap (molecular conditions) | +0.04 | **−0.02** | **no relationship** |
| λ_max ↔ gap (`hydro`) | −0.20 | **−0.48** | **real, and inverted** |
| `fg_legacy` ↔ `fg_penult` | **+0.95** | — | **one signal, not two** |

Three consequences:

1. **Activation energy is not a curvature proxy.** Large learning rates find flatter
   minima *and* raise activation energy (ρ(lr, `fg_legacy`) ≈ +0.5, ρ(lr, λ_max)
   ≈ −0.4 to −0.8). Given the learning rate, activation energy carries almost no
   curvature information in either direction.
2. **But it tracks generalisation better than exact curvature does.** On every
   molecular condition λ_max is *precisely uninformative* about the gap (ρ ≈ 0.00 at
   n ≈ 1,000–1,400) while activation energy carries +0.16 to +0.35 — and unlike (1)
   this *strengthens* under the lr control. The 30x-more-expensive, theoretically
   motivated quantity is beaten at its own job by the forward-only heuristic it was
   supposed to justify. It is still far too weak to convert into selection.
3. **The two proxies are one signal**, so their agreement anywhere is not independent
   corroboration.


### 13.1 Is λ_max simply the wrong curvature summary?

Andriushchenko et al. (ICML 2023) report that sharpness tracks the learning rate
better than it tracks generalisation, which §13 reproduces for λ_max. That invites an
obvious rescue: λ_max is **one direction out of millions**, so perhaps bulk curvature
would do better. We ran the Hessian **trace** (Hutchinson, 30 Rademacher probes,
exact HVPs) alongside λ_max on three conditions — 720 models, 15 shards — and formed
an approximation to Petzka et al.'s (NeurIPS 2021) **relative flatness** as
`‖w‖²·tr(H)` from quantities already stored. Full tables: `RESULTS_trace.md`.

The rescue does not work, and λ_max really is a small part of the picture: it is a
median **0.12–0.14** of the trace, so it summarises roughly an eighth of the total
curvature. Measuring the other seven eighths does not help.

""")
    _trace_table(L)
    L.append("""
`gap` = test − train, so a positive ρ is the direction flatness theory predicts and a
negative one is that theory inverted.

**Three findings, and the third is the one that matters.**

1. **Bulk curvature is no better than the top eigenvalue.** tr(H) is null on
   `caco2_scaffold` (−0.02), weakly inverted on `cond7` (−0.17), and strongly
   inverted on `hydro` (−0.52) — the same place λ_max inverts. The failure in §13 is
   not an artifact of summarising curvature by one direction.

2. **Relative flatness looks like the exception — and is not.** It is the only
   curvature-derived quantity here that behaves as flatness theory predicts on the
   molecular conditions. But it is a product of two factors, and the decomposition
   shows the **weight norm** carries it: ‖w‖² alone already reaches +0.18 and +0.31,
   and once ‖w‖² is controlled the trace contributes **+0.08 and +0.07**. On `hydro`
   the decomposition inverts — there curvature does the work (−0.49) and the norm
   does not. Same measure, opposite internals, decided by the condition.

3. **This corrects the proxy's motivation rather than the proxy.** Activation energy
   is an *inverse* measure of bulk curvature (partial ρ to tr(H) = −0.44, −0.61,
   −0.21) and carries almost no λ_max information once the learning rate is
   controlled. But on the molecular conditions it is substantially the *same signal*
   as relative flatness (ρ = +0.44 and **+0.72**) — that is, the same signal as the
   **norm** half of relative flatness. So `fg_legacy` was never a cheap sharpness
   surrogate; where it behaves like anything principled it behaves like a norm-based
   capacity measure. That is a motivation this study can actually support, and it is
   not the one the method was built on.

**None of it selects.** On the same pools, `hess_trace` and `‖w‖²·tr(H)` are worse
than validation on all three conditions (0/5 runs on `cond7` and `hydro`; the best
either manages is 3/2 on `caco2_scaffold` at n = 5). The trace also costs **1.9–3.0x
a power iteration**, so the better curvature summary is both more expensive and no
more useful — which is the cleanest statement of the negative result in this file.

## 14. The degeneracy trap

Full audit in `DEGENERACY_AUDIT.md`. **All three signals are minimised by a network
that does nothing** — a dead network has zero activation energy and zero curvature. On
a condition where predicting the training mean is a decent OOD strategy, every proxy
selects the same degenerate model and appears to reach oracle performance.

| condition | constant predictors | `fg_legacy` picks one in |
|---|---:|---:|
| `amylase` | **24–28%** | **80% of runs** |
| every other condition | **0.0%** | 0% |

`amylase` initially showed −1.39 MSE, 20/20 runs, p < 0.001. It is an artifact, and
`amylase` is therefore excluded from every proxy claim in this file. FLIP2's own
authors independently report that *"supervision decreased performance on every Amylase
split"* — our median test Spearman there is −0.011.

`hydro` has many *partially* dead networks (~30% with >50% dead units) but **no**
constant predictors, which is why its results are reported rather than excluded.

**Any proxy paper that does not run this check cannot distinguish "my signal works"
from "my signal found a dead network."**

## 15. Are the models any good?

A negative result only counts if the models are competent. Raw-scale comparison to
published values (`sanity_checks.md`):

| dataset | ours | published | verdict |
|---|---|---|---|
| Lipophilicity | MAE **0.528** | TDC leaderboard 0.511 | within 3% |
| Caco2-Wang | MAE **0.306** | TDC leaderboard 0.269–0.288 | within 7–14% |
| FLIP2 Hydro | 24,935 variants, split at −3.206 | 24,935, stated median −3.21 | **exact reproduction** |
| FLIP2 Amylase | 3,706 variants | 3,706 | **exact reproduction** |
| GDSC2 | RMSE **1.535** | leave-drug-out 0.6–2.1 | inside the band |

From a plain MLP on Morgan fingerprints with no pretraining. **The negative results
are not an artifact of weak models.**""")



def metric_robustness(L):
    """Do the conclusions survive a change of DEPLOYMENT metric?

    Every selection rule is left untouched (all still rank by validation MSE); only
    the metric used to score the deployed model changes. MSE is what this study
    optimises, but the TDC leaderboards report MAE and FLIP2 reports Spearman -- so a
    conclusion that holds only under MSE is a conclusion about our scoring choice.
    """
    SRC = {"caco2_random": P1A, "caco2_scaffold": P1A, "lipo_random": P1A,
           "lipo_scaffold": P1A, "cond7": P1A, "amylase": P1A, "hydro": P1A,
           "gdsc_drug": GD}
    SELS = [("val", None), ("fg_legacy", "fg_legacy"), ("fg_penult", "fg_penult"),
            ("hess_top", "hess_top"), ("val+legacy", "+fg_legacy"),
            ("val+penult", "+fg_penult"), ("train_mse", "__tr"), ("n_params", "__np")]
    METRICS = [("test_mse_std", False, "MSE (ours)"),
               ("test_mae_raw", False, "MAE (TDC)"),
               ("test_spearman", True, "Spearman (FLIP2)")]

    def sel_idx(k, s):
        if s is None: return k.val_mse_std.idxmin()
        if s == "__tr": return k.train_mse_std.idxmin()
        if s == "__np": return k.n_params.idxmin()
        if s.startswith("+"):
            c = s[1:]; o = k[k[c].notna()]
            return o.index[int(np.argmin(rankdata(o.val_mse_std) + rankdata(o[c])))]
        o = k[k[s].notna()]; return o[s].idxmin()

    L.append("""
## 16. Does the conclusion depend on the deployment metric?

Selection rules are **unchanged** — every selector still ranks by validation MSE.
Only the metric used to score the deployed model varies. This matters because MSE is
our choice: the TDC leaderboards report **MAE**, and FLIP2 reports **Spearman**. A
result that holds only under MSE is a statement about our scoring, not about selection.
""")
    L.append("| condition | under MSE (ours) | under MAE (TDC) | under Spearman (FLIP2) |")
    L.append("|---|---|---|---|")
    for cond, root in SRC.items():
        R = runs(root, cond)
        if len(R) < 5: continue
        cells = []
        for mcol, hib, _ in METRICS:
            base = np.array([R[i].loc[sel_idx(R[i], None), mcol] for i in range(len(R))])
            ps, names = [], []
            for nm, sp in SELS:
                if nm == "val": continue
                v = np.array([R[i].loc[sel_idx(R[i], sp), mcol] for i in range(len(R))])
                d = (base - v) if hib else (v - base)
                d = d[np.isfinite(d)]
                ps.append(float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0)
                names.append((nm, float(np.median(d))))
            adj = holm(np.array(ps))
            win = [nm for (nm, md), a in zip(names, adj) if a < 0.05 and md < 0]
            cells.append(", ".join(f"`{w}`" for w in win) or "—")
        L.append(f"| `{cond}` | {cells[0]} | {cells[1]} | {cells[2]} |")
    L.append("""
**In the shared pool, the proxy wins are metric-fragile.** `val+legacy` and
`val+penult` beat validation on `cond7` under MSE and **both disappear under MAE and
under Spearman**. They were already practically negligible (Δ = 0.0004 on a scale of
0.49); they are also metric-fragile.

**The simple baselines are robust.** `n_params` beats validation on `hydro` under all
three metrics, and `train_mse` additionally wins there under MAE.

### 16.1 The same question for Phase 1B

Selection is not the only place a signal can act — it can also steer the *search*.
Applying the identical treatment to the eight sequential conditions (deploy rule
fixed, only the scoring metric varied, Holm over the 8 contrasts within each
condition and metric):
""")
    B1SRC = [("cond7", "artifacts/results/phase1b_sequential/runs"),
             ("hydro", "artifacts/results/phase1b_sequential/runs"),
             ("gdsc_drug", "artifacts/results/exploratory/gdsc/phase1b/runs"),
             ("caco2_random", "artifacts/results/exploratory/phase1b_extra/runs"),
             ("caco2_scaffold", "artifacts/results/exploratory/phase1b_extra/runs"),
             ("lipo_random", "artifacts/results/exploratory/phase1b_extra/runs"),
             ("lipo_scaffold", "artifacts/results/exploratory/phase1b_extra/runs"),
             ("amylase", "artifacts/results/exploratory/phase1b_extra/runs")]
    BM = ["random_val", "tpe_val", "hebo_val", "tpe_fg_mo", "hebo_fg_mo",
          "tpe_penult_mo", "hebo_penult_mo", "tpe_hess_mo", "hebo_hess_mo"]
    BSIG = {"tpe_fg_mo": "fg_legacy", "hebo_fg_mo": "fg_legacy",
            "tpe_penult_mo": "fg_penult", "hebo_penult_mo": "fg_penult",
            "tpe_hess_mo": "hess_top", "hebo_hess_mo": "hess_top"}

    def bdep(g, sig, col):
        ok = g[(g.status_train == "ok") & g.val_mse_std.notna()]
        if not len(ok): return np.nan
        if sig is None: return float(ok.loc[ok.val_mse_std.idxmin(), col])
        o = ok[ok[sig].notna()]
        if not len(o): return np.nan
        j = o.index[int(np.argmin(rankdata(o.val_mse_std) + rankdata(o[sig])))]
        return float(o.loc[j, col])

    L.append("| condition | under MSE (ours) | under MAE (TDC) | under Spearman (FLIP2) |")
    L.append("|---|---|---|---|")
    for cond, root in B1SRC:
        cells = []
        for mcol, hib, _ in METRICS:
            V = {m: np.array([bdep(pd.read_parquet(f), BSIG.get(m), mcol)
                              for f in sorted(glob.glob(f"{root}/{m}/{cond}/*.parquet"))])
                 for m in BM}
            if not len(V.get("tpe_val", [])): cells.append("—"); continue
            base = V["tpe_val"]; ps, names = [], []
            for m in BM:
                if m == "tpe_val" or not len(V[m]): continue
                k = min(len(V[m]), len(base))
                d = (base[:k] - V[m][:k]) if hib else (V[m][:k] - base[:k])
                d = d[np.isfinite(d)]
                if not len(d): continue
                ps.append(float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0)
                names.append((m, float(np.median(d))))
            if not ps: cells.append("—"); continue
            adj = holm(np.array(ps))
            win = [nm for (nm, md), a in zip(names, adj) if a < 0.05 and md < 0]
            cells.append(", ".join(f"`{w}`" for w in win) or "—")
        flag = " ⚠ degenerate" if cond == "amylase" else ""
        L.append(f"| `{cond}`{flag} | {cells[0]} | {cells[1]} | {cells[2]} |")
    L.append("""
`amylase` ⚠ is the degeneracy artifact of section 14, not a result: its 1B
trajectories are not filtered (see the note in section 10), so the arms that "win"
there are the ones that found a constant predictor. Discount that row.

**This is where the study's one metric-robust proxy result lives.** On
`lipo_scaffold`, proxy-steered *search* Holm-beats validation-driven search under all
three metrics, and `tpe_penult_mo` survives in every one of them — the only selector
or method anywhere in this study to do so. Four arms win under MSE, two under MAE,
one under Spearman, so the effect narrows as the metric changes but does not vanish
the way the `cond7` shared-pool win does.

Three things keep this from being a positive headline. It is **one condition of
eight** — the matching random split, `lipo_random`, shows nothing under any metric, and
`caco2_scaffold` shows nothing either, so this is not a general property of scaffold
shift. It was **post-hoc**: `lipo_scaffold` was not in the pre-registered Phase 1B
pair, and it is the condition that motivated running 1B on everything else. And the
same signal family is Holm-*worse* on `gdsc_drug`. Reported as what it is: a real,
metric-robust, single-condition effect inside an otherwise negative study — the kind
of result that is worth a paragraph and not a title.
""")


def _trace_table(L):
    """The §13.1 correlation table, computed from the trace tree rather than typed.

    Hardcoding these numbers is the exact failure that dropped 6 of 8 Phase 1B
    conditions from this file earlier: a static table survives the data changing
    underneath it. If the trace run is extended, this table follows.
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from analyze_trace import load as _tload, partial as _tpart
    except Exception:
        L.append("*(trace tree unavailable — run `scripts_icbinb/analyze_trace.py`)*")
        return
    CONDS = ["caco2_scaffold", "cond7", "hydro"]
    data = {}
    for c in CONDS:
        R = _tload(c)
        if R:
            d = pd.concat(R)
            data[c] = d.assign(w2=d.weight_norm_fro**2)
    if not data:
        L.append("*(trace tree unavailable)*"); return
    ROWS = [("λ_max ~ gap", "hess_top", None),
            ("**tr(H)** ~ gap", "hess_trace", None),
            ("‖w‖²·tr(H) ~ gap *(relative flatness, approx.)*", "petzka", None),
            ("‖w‖² alone ~ gap", "w2", None),
            ("tr(H) ~ gap, **also** controlling ‖w‖²", "hess_trace",
             ("lr", "width", "depth", "w2"))]
    L.append("| relationship (partial ρ given lr, width, depth) | "
             + " | ".join(f"`{c}`" for c in data) + " |")
    L.append("|---" + "|---:" * len(data) + "|")
    for lab, col, ctrl in ROWS:
        cells = []
        for c, d in data.items():
            _, pr, _ = (_tpart(d, col, "gap", ctrl) if ctrl else _tpart(d, col, "gap"))
            txt = f"{pr:+.2f}".replace("-", "−")
            cells.append(f"**{txt}**" if abs(pr) >= 0.20 else txt)
        L.append(f"| {lab} | " + " | ".join(cells) + " |")


def coverage(L):
    """Walk the results tree and state where every experiment family is reported.

    This exists because the failure it guards against already happened twice: a
    hardcoded condition list silently dropped 6 of 8 Phase 1B conditions from this
    file, and both featshift arms from the phase1a table. Anything on disk that is not
    in REPORTED below is flagged UNREPORTED rather than quietly omitted.
    """
    REPORTED = {
        "phase1a_pool/candidates":
            ("shared 48-model pool, 7 pre-registered conditions",
             "§7 - `RESULTS_BY_DATASET.md` §1"),
        "exploratory/gdsc":
            ("GDSC2 drug-holdout, 1A + 1B",
             "§7, §10, §16.1 - `RESULTS_BY_DATASET.md` §1, §4"),
        "exploratory/featshift":
            ("engineered extreme feature shift, 1A", "§7 - `RESULTS_BY_DATASET.md` §1"),
        "exploratory/shift_severity":
            ("synthetic alpha severity knob, 10 conditions",
             "§9 - `RESULTS_BY_DATASET.md` §1"),
        "ablations/fixed_arch":
            ("architecture pinned to 256x2", "§8 - `RESULTS_BY_DATASET.md` §2"),
        "ablations/valfree_pool":
            ("validation folded into training (n_val = 0)",
             "§11 - `RESULTS_BY_DATASET.md` §3"),
        "phase1b_sequential/runs":
            ("pre-registered sequential HPO, cond7 + hydro",
             "§10 - `RESULTS_BY_DATASET.md` §4"),
        "exploratory/phase1b_extra":
            ("sequential HPO on the 5 remaining conditions",
             "§10 (summary), §16.1 - `RESULTS_BY_DATASET.md` §4"),
        "ablations/valfree_sequential":
            ("sequential search with no validation at all",
             "§12 - `RESULTS_BY_DATASET.md` §5"),
        "exploratory/trace":
            ("Hessian trace + relative flatness",
             "§13.1 - `RESULTS_BY_DATASET.md` §6 - `RESULTS_trace.md`"),
    }
    counts = {}
    for dirpath, _, files in os.walk("artifacts/results"):
        n = len([f for f in files if f.endswith(".parquet")])
        if not n: continue
        parts = dirpath.split(os.sep)
        key = "/".join(parts[1:3]) if len(parts) > 2 else "/".join(parts[1:])
        counts[key] = counts.get(key, 0) + n
    L.append("\n## 18. Coverage — every experiment run, and where it is reported\n")
    L.append("Generated by walking `artifacts/results/` at build time, so a tree that "
             "exists on disk but is reported nowhere appears here as **UNREPORTED** "
             "instead of being silently omitted.\n")
    L.append("| experiment family | shards | what it is | reported in |")
    L.append("|---|---:|---|---|")
    total, unrep = 0, []
    for k in sorted(counts):
        total += counts[k]
        if k in REPORTED:
            what, where = REPORTED[k]
            L.append(f"| `{k}` | {counts[k]} | {what} | {where} |")
        else:
            unrep.append(k)
            L.append(f"| `{k}` | {counts[k]} | — | **UNREPORTED** |")
    L.append(f"\n**{total} shards across {len(counts)} families.** "
             + ("Every family is reported in this file, in `RESULTS_BY_DATASET.md`, "
                "or in the companion noted above."
                if not unrep else
                f"⚠ **{len(unrep)} unreported:** " + ", ".join(f"`{u}`" for u in unrep)))
    L.append("\n**The two files divide the work deliberately.** This file is the "
             "*argument* — one row per condition, aggregated, Holm-corrected, in "
             "train-σ standardised MSE. `RESULTS_BY_DATASET.md` is the *evidence* — one "
             "table per condition per family, every selector scored under MAE and "
             "Spearman as well, which is what the TDC leaderboards and FLIP2 actually "
             "report. Read this file for what is claimed and `RESULTS_BY_DATASET.md` to "
             "check it against the field's own metrics.\n")
    L.append("Analyses that live only in a companion file, each summarised here: "
             "`RESULTS_signals.md` (full partial-correlation tables and within-lr-bin "
             "checks, §13), `RESULTS_phase1b.md` (the five-panel deploy-rule "
             "decomposition, §10), `DEGENERACY_AUDIT.md` (§14), `sanity_checks.md` "
             "(raw-scale and leaderboard comparison, §15), `RESULTS_trace.md` (§13.1), "
             "plus `RESULTS_ablations.md` and `RESULTS_exploratory.md`, the per-family "
             "long forms of §8, §9, §11 and §12.")


def main():
    L=["# RESULTS — master sweep (all completed experiments)",""]
    L.append("*Auto-generated by `scripts_icbinb/sweep_all.py` — do not hand-edit; "
             "edit the script.*")
    L.append(r"""
## 1. What question this answers

Choosing a model — architecture, learning rate, regularisation — normally means
holding out a **validation split** and keeping the configuration with the lowest
validation loss. In biology that split is often drawn from the same batches, donors,
assays or chemical series as the training data, while deployment is not. The
validation set is in-distribution; the thing you deploy on is not.

A recurring proposal is to select on **geometry of the training loss surface**
instead — flat minima are supposed to generalise — which needs no held-out data at
all. This sweep asks one question across every dataset we have:

> **Can a training-only signal replace, or usefully augment, a validation set when
> the deployment distribution differs from the training one?**

## 2. How one cell is produced

Every cell in every table below is built the same way, so cells are comparable.

1. **Candidate pool.** 48 hyperparameter configurations from a scrambled Latin
   hypercube (learning rate, weight decay with 20% exact zeros, dropout, width,
   depth). *Every selector ranks the same 48 models*, so comparisons are paired and
   no selector gets a different search.
2. **Train.** Each configuration is trained once — MLP, ReLU, dropout, **no BatchNorm**, Adam,
   batch 64, **100 fixed epochs, no early stopping** (early stopping is itself a
   validation-based decision and would contaminate the comparison).
3. **Outer runs.** The whole thing repeats over 10–30 independent outer runs. The
   `reps` column reports how many. This is the replication unit of every statistic
   below and is defined precisely in §2.1.
4. **Filter.** Degenerate models are removed — constant predictors
   (`pred_std_test` < 1e-6) and models with >50% dead ReLU units. `kept` shows how
   many of the 48 survive. This matters: all three signals are *minimised* by a
   network that does nothing, so leaving degenerates in lets a dead network
   masquerade as a discovery.
5. **Select and deploy.** Each selector picks one of the surviving models using
   **training and validation information only**, and we then report that model's
   **test** loss. The test set is never consulted by any selector — only by the
   `oracle` row, which exists to show what was achievable.
6. **Compare.** Paired within outer run, Wilcoxon signed-rank, against two
   references (§3).

Test losses are on **train-σ standardised units** (targets divided by the training
standard deviation), so they are comparable within a condition but not across them.
Raw-scale RMSE/MAE and the comparison to published leaderboards are in
`sanity_checks.md`.

### 2.1 What an "outer run" is

Not 10 random initialisations. An outer run is a **complete, independent replication
of the entire pipeline**, indexed by an integer `r`, from which six seeds are derived
deterministically as `10000·r + 1 … 10000·r + 6`:

| seed | what it randomises |
|---|---|
| `split_seed` | the train / validation / test partition |
| `hp_seed` | **the 48-candidate Latin hypercube itself** |
| `init_seed` | weight initialisation (offset by candidate id, so candidates differ too) |
| `loader_seed` | mini-batch composition and ordering |
| `proxy_seed` | the stochastic proxy estimators (Hutchinson probes, power-iteration start) |
| `sampler_seed` | the HPO sampler's own randomness (Phase 1B only) |

Two consequences worth stating, because they change what the spread means.

**The candidate pool is redrawn every outer run.** "Every selector ranks the same 48
models" holds *within* an outer run — which is what makes the comparisons paired —
but the 48 configurations themselves differ across runs. So no result here is a
property of one lucky Latin hypercube draw, and the run-to-run spread includes the
variation from resampling the search space.

**The test set is redrawn too, on all but the FLIP2 conditions.** Because
`split_seed` varies, `caco2_*`, `lipo_*`, `cond7`, `gdsc_drug`, `shiftsev_*` and
`featshift_*` get a fresh train/val/test partition each run, so their spread includes
test-set sampling noise. The two FLIP2 conditions, `amylase` and `hydro`, do **not**:
their split is a deterministic cut (position-based and fitness-median respectively),
so all 30 runs score the identical 12,468-variant test set and their spread reflects
only training stochasticity and the pool redraw. This is why FLIP2 IQRs are not
comparable to molecular IQRs, and it is also what makes the FLIP2 numbers directly
comparable to FLIP2's published ones.

Pairing works across experiment trees because run `r` means the same thing
everywhere: the `valfree_pool` comparison in §11 pairs run-for-run against `phase1a`
and therefore holds the split, the pool and the initialisation fixed while changing
only whether the validation rows are trained on.

### 2.2 Dispersion — both conventions, and what each is for

Numeric cells read `median [25th, 75th] · mean ± sd` over outer runs (sample sd,
ddof = 1). Two summaries are given because the field and this design want different
things.

**`mean ± sd` is for comparability.** The TDC leaderboards report mean ± sd over 5
seeds and the ML-for-biology literature reads that form by default, so refusing to
print it would just make this work harder to place next to published numbers.

**The IQR is what this design actually warrants.** Every contrast here is **paired
within an outer run** — same split, same 48 candidates, same initialisation — and a
standard deviation invites the unpaired comparison that throws that pairing away.
With 10–30 runs of a non-normal, heavily tied distribution, the paired Wilcoxon
signed-rank is both more powerful and less assumption-laden than anything the sd
supports.

**Neither is the inferential claim.** Both dispersions are descriptive; the `p` and
`p_holm` columns carry the inference. Two consequences to read carefully:

* Where two selectors' intervals **overlap almost entirely and `p` is still small**,
  that is the pairing doing the work, not an inconsistency. `phase1a/hydro` is the
  clearest case: `val` and `val+legacy` overlap heavily on both summaries, yet the
  paired test is decisive at 5/20 runs, because the same 48 models are being ranked
  two ways inside each run.
* Where the **mean sits well away from the median**, the distribution is skewed by a
  few bad runs and the median is the more faithful summary of a typical deployment.
  This is common on the molecular conditions, whose test set is resampled every run
  (§2.1), so an unlucky split inflates the mean without moving the median.

## 3. The two reference points — and why both are needed

**`val`** — what a practitioner actually does. Beating it is the practical claim.

**`random(E)`** — the *expected* test loss of picking a configuration uniformly at
random, computed as the **mean** test loss over the pool. It is the "do nothing"
baseline: the cost of closing your eyes and grabbing one.

We use the mean rather than one sampled draw deliberately — it is E[loss] under a
uniform pick, so it is deterministic and adds no sampling noise of its own.

**Why `random(E)` must be reported.** Validation-versus-proxy is a *relative*
comparison, and a proxy can beat validation while both are worse than not choosing at
all. That is not hypothetical: on `fixed_arch/hydro` below, the proxies beat `val`
while `val` is itself significantly *worse than random* — so the "win" is against a
broken reference, not evidence of signal. **A selector that does not clear
`random(E)` carries no usable signal, whatever its p-value against `val`.**

`val_fail ρ` is Spearman(validation loss, test loss) across the pool, median over
outer runs: +1 means validation ranks candidates exactly as deployment would, ≈0
means validation ranking is uninformative, negative means actively misleading.

## 4. `p` versus `p_holm`

Both columns are reported throughout. They answer different questions.

**Raw `p`** — the probability of seeing a difference at least this large *if this one
selector were truly identical to the reference*. Correct for a single hypothesis
fixed in advance.

**`p_holm`** — the same evidence adjusted for the fact that we tested **m** selectors
in this condition and are reporting whichever looked best.

### The math

Given raw p-values $p_1,\dots,p_m$, sort them ascending as $p_{(1)} \le \dots \le p_{(m)}$:

$$\tilde p_{(k)} \;=\; \min\!\Big(1,\; \max_{j \le k} \big[(m-j+1)\, p_{(j)}\big]\Big)$$

The $k$-th smallest is multiplied by $(m-k+1)$ — the smallest gets the harshest
penalty $(\times m)$, the largest the gentlest $(\times 1)$ — and the running maximum
enforces monotonicity so an adjusted value never falls below a smaller one. Reject
while $\tilde p < \alpha$.

This controls the **family-wise error rate**: $P(\text{any false positive}) \le \alpha$
across the whole family, rather than per test. Holm is uniformly more powerful than
Bonferroni (which multiplies *every* p by $m$) with an identical guarantee.

### The intuition

With $m$ independent tests at $\alpha = 0.05$, the chance of at least one false
positive is $1 - 0.95^{m}$ — for $m = 8$ that is **34%**, not 5%. Scanning a list of
selectors and reporting the winners is exactly that situation.

Simulating 8 selectors that are **all genuinely useless** (pure noise), running this
analysis 4,000 times:

| decision rule | false discovery in | bogus findings per study |
|---|---:|---:|
| raw `p < 0.05` | **31.3%** of studies | 0.37 |
| `p_holm < 0.05` | **4.2%** of studies | 0.04 |

### Where it changed a conclusion here

Phase 1B on `hydro`, 8 methods against `tpe_val`: `hebo_fg_mo` (raw p = 0.014),
`tpe_fg_mo` (0.020) and `hebo_penult_mo` (0.037) all look significantly *worse* than
validation-driven search. After Holm they become 0.109, 0.137 and 0.223 — **none
survive**. That is the difference between claiming *"proxy-driven search actively
hurts under extrapolation"* and correctly stating *"we could not establish that it
hurts."* Three hits at p ≈ 0.01–0.04 out of 8 tests is close to what pure noise
produces.

**Caveat, stated honestly.** Holm is conservative and can mask a real effect. Failing
to survive it means *not established*, not *shown absent*. Columns below therefore
report which selectors survive Holm; the underlying raw p-values live in the
per-phase files.


## 5. Naming — what every name means

Names in this project are terse and several are historical. This is the full key.

### 5.1 The three signals

All are computed on **training data only** — that is the whole point; a signal that
needs held-out data is not a replacement for held-out data.

| name | meaning | definition |
|---|---|---|
| `fg_legacy` | per-neuron **activation energy**, averaged over all hidden layers | $\max_{\text{batch}} \operatorname{mean}_\ell \lVert A_\ell\rVert_F^2 / (B\cdot w)$ for post-activation $A_\ell$, batch $B$, width $w$ |
| `fg_penult` | the same quantity read off the **penultimate** (last hidden) layer only | as above but $\ell$ = last hidden layer — the representation the output head actually sees |
| `hess_top` | **top Hessian eigenvalue** $\lambda_{\max}$ of the training loss — true curvature | power iteration with exact Hessian-vector products (Pearlmutter double-backward) |

Etymology, since the names are not self-explanatory:

* **`fg`** is historical — it comes from this repository's original name (`FGBO`), not
  from anything meaningful about the quantity. Read it as "the activation-energy
  family".
* **`legacy`** means *the original formula* (depth-averaged), **not** "deprecated".
  It is the primary signal under test.
* **`penult`** = penultimate layer.
* **`hess_top`** and **$\lambda_{\max}$** are used interchangeably in prose.

> ⚠ **Do not call all three "flatness" signals.** Only `hess_top` measures curvature.
> `fg_legacy` and `fg_penult` measure **activation energy**, which is merely
> *motivated* as a cheap stand-in for sharpness — and this study shows that
> motivation is unfounded: conditioned on learning rate, activation energy carries
> almost no curvature information (partial ρ = −0.09 to −0.14; see
> `RESULTS_signals.md` §2). "Train-only geometry signals" is the accurate collective
> term.

`hess_top` is a **control, not a competitor**. Activation energy is motivated as a
cheap stand-in for sharpness; running exact curvature alongside it everywhere turns
that motivation into something measurable. It is ~30x more expensive, which is why
nobody uses it in practice.

### 5.2 Selectors

A **selector** is a rule that picks one model from the pool using training and
validation information only. Naming convention: **`A+B` means rank-sum of A and B** —
rank every candidate by A, rank by B, add the ranks, take the minimum. Rank-sum is
scale-free; a product is dominated by whichever quantity has the larger dynamic range,
and a weighted sum needs a weight we would have had to tune, which would quietly
reintroduce the validation set we are trying to eliminate.

| selector | picks the model with lowest… | why it is here |
|---|---|---|
| `val` | validation MSE | what a practitioner actually does — the reference |
| `train_mse` | training MSE | the val-free alternative; also a strong baseline |
| `fg_legacy` / `fg_penult` / `hess_top` | that signal alone | pure train-only selection |
| `val+legacy` | rank(val) + rank(`fg_legacy`) | *augment* validation rather than replace it |
| `val+penult` / `val+hess` | rank(val) + rank(that signal) | same, other signals |
| `train+legacy` etc. | rank(train MSE) + rank(signal) | the val-free ablations, where no validation exists |
| `n_params` | fewest parameters | capacity heuristic with no geometry in it — a deliberately dumb baseline |
| `random(E)` | *(not a selector)* mean test loss over the pool | the "do nothing" bar; see §3 |
| `oracle` | lowest **test** MSE | not achievable — shows what was on the table |

`oracle` and `random(E)` are the only rows that touch the test set; every real
selector is blind to it.

### 5.3 Conditions (datasets × split)

| condition | what it is |
|---|---|
| `caco2_random`, `caco2_scaffold` | TDC Caco2-Wang permeability; random vs Bemis–Murcko scaffold split |
| `lipo_random`, `lipo_scaffold` | TDC Lipophilicity; same two split types |
| **`cond7`** | "**condition 7**" from the pre-registration — a *constructed* split: take the scaffold split, merge train+val, re-split those **randomly**, leave test scaffolds untouched. Validation becomes ID-like while test stays OOD. The named hypothesis condition. |
| `amylase` | FLIP2 amylase **close→far**: train on mutations near the active site, test on distal ones — a *position* shift |
| `hydro` | FLIP2 hydrophobic-core (SH3 domain) **low→high**: train below median fitness, test above — *fitness extrapolation* |
| `gdsc_drug` | TDC GDSC2 drug response, holding out whole **compounds** — natural ID-val / OOD-test |
| `shiftsev_{caco2,lipo}_a000…a100` | synthetic severity knob; the digits are **alpha x 100** (`a075` = α = 0.75). α = 0 is an IID split, α = 1 is pure feature extrapolation |

### 5.4 Phase 1B method names

Pattern: **`{sampler}_{objective}`**.

* sampler — `random` (uniform), `tpe` (Optuna TPE), `hebo` (HEBO)
* objective — `val` = single-objective on validation MSE; `*_mo` = **multi-objective**
  on (validation MSE, signal), where `fg` = `fg_legacy`, `penult` = `fg_penult`,
  `hess` = `hess_top`

So `hebo_penult_mo` = HEBO optimising validation MSE and `fg_penult` jointly.

The three `_mo` signal families in full:

| arm | jointly optimises | second objective is |
|---|---|---|
| `tpe_fg_mo`, `hebo_fg_mo` | (val MSE, `fg_legacy`) | activation energy, all hidden layers |
| `tpe_penult_mo`, `hebo_penult_mo` | (val MSE, `fg_penult`) | activation energy, penultimate layer |
| `tpe_hess_mo`, `hebo_hess_mo` | (val MSE, `hess_top`) | $\lambda_{\max}$, true curvature |

TPE arms use Optuna **MOTPE** (non-dominated sorting, hypervolume tie-break); HEBO
arms use `GeneralBO(num_obj=2)` — the plain `HEBO` class has no `num_obj`, so
multi-objective requires the general optimiser. Neither scalarises, so no weight had
to be chosen. Deployment then applies the within-budget rank-sum over observed trials.

> ⚠ **Name collision — the same six names mean different methods in different trees.**
> Under `phase1b_sequential/` the first objective is **validation** MSE
> (`base = "val_mse_std"`, `n_val` > 0). Under `ablations/valfree_sequential/` it is
> **training** MSE (`base = "train_mse_std"`, `n_val` = 0 — validation is folded into
> training and no validation set exists). So `tpe_fg_mo` optimises
> (val MSE, `fg_legacy`) in one place and (train MSE, `fg_legacy`) in the other.
> Always read the method name together with its experiment directory.

### 5.5 Experiments

| name | what it varies |
|---|---|
| `phase1a` | **shared pool** — every selector ranks the same 48 models. Isolates *selection* from *search*. |
| `phase1b` | **sequential HPO** — each method runs its own 48-trial search. Asks whether a signal can *drive* the search. |
| `fixed_arch` | architecture pinned to 256x2; only lr/weight-decay/dropout vary. Tests whether failure is a cross-architecture comparability problem. |
| `valfree_pool` | validation folded into training (`n_val = 0`); only train-side selectors exist. Is validation data better spent as training data? |
| `valfree_sequential` | the six multi-objective methods searching `(train_mse, signal)` — no validation anywhere |
| `shift_severity` | the synthetic α knob of §5.3 |

### 5.6 Column and suffix conventions

| token | meaning |
|---|---|
| `_std` | on **train-σ standardised** targets (divided by the training standard deviation) — comparable within a condition, not across |
| `_raw` | original target units (log-permeability, logD, fitness, ln IC50) |
| `_e25`, `_e50` | the same quantity measured at epoch 25 / 50 instead of 100 |
| `reps` | number of independent outer runs (different split/init/order seeds) |
| `kept` | candidates surviving the degeneracy filter, out of 48 |
| `val_fail ρ` | Spearman(validation loss, test loss) across the pool — +1 = validation ranks as deployment would, ≈0 = uninformative, negative = misleading |
| `gap` | generalisation gap = test MSE − train MSE |
| `dead_relu_frac` | fraction of ReLU units that never activate |
| `pred_std_test` | standard deviation of the model's test predictions; `< 1e-6` = a **constant predictor** (degenerate) |
| `random(E)` | the **E** is for *expectation* — E[loss] under a uniform random pick |

## 6. Reading the tables

`beats val (Holm)` and `beats random (Holm)` list only selectors that are both
**better** (negative median paired difference) **and** Holm-significant at 0.05
within that condition's family. A dash means nothing cleared the bar.
""")
    beats_val, beats_rand, nothing = [], [], []
    for exp, blurb, entries in POOLS:
        L.append(f"\n## {7+[e[0] for e in POOLS].index(exp)}. `{exp}`\n")
        L.append(blurb + ".\n")
        L.append("| condition | reps | kept | val_fail ρ | random(E): med [IQR] · mean ± sd | val: med [IQR] · mean ± sd | val vs random | "
                 "beats val (Holm) | beats random (Holm) | note |")
        L.append("|---|---:|---:|---:|---:|---:|---|---|---|---|")
        for cond, root, note in entries:
            R=runs(root,cond)
            if len(R)<5: continue
            V={n:np.array([pick(k,s) for k in R]) for n,s in SEL}
            V["random(E)"]=np.array([pick(k,"__rand") for k in R])
            rho=float(np.median([spearmanr(k.val_mse_std,k.test_mse_std).statistic for k in R]))
            res={}
            for ref in ("val","random(E)"):
                names=[n for n,_ in SEL if n!=ref]+(["random(E)"] if ref=="val" else [])
                ps,md=[],[]
                for n in names:
                    d=V[n]-V[ref]; d=d[np.isfinite(d)]
                    ps.append(float(wilcoxon(d).pvalue) if len(d)>2 and np.any(d!=0) else 1.0)
                    md.append(float(np.median(d)) if len(d) else np.nan)
                adj=holm(np.array(ps))
                res[ref]=[n for n,m,a in zip(names,md,adj) if a<0.05 and m<0]
            dv=V["val"]-V["random(E)"]
            pv=float(wilcoxon(dv).pvalue) if np.any(dv!=0) else 1.0
            vr=("**beats random**" if np.median(dv)<0 and pv<0.05 else
                "**WORSE than random**" if np.median(dv)>0 and pv<0.05 else "no better than random")
            bv=", ".join(f"`{x}`" for x in res["val"] if x!="random(E)") or "—"
            br=", ".join(f"`{x}`" for x in res["random(E)"] if x!="val") or "—"
            label = cond.replace("shiftsev_","").replace("_a"," α=") if cond.startswith("shiftsev_") else cond
            if cond.startswith("shiftsev_"):
                ds, al = cond.replace("shiftsev_","").split("_a"); label = f"{ds} α={int(al)/100:.2f}"
            L.append(f"| `{label}` | {len(R)} | {np.median([len(k) for k in R]):.0f}/48 | {rho:+.3f} | "
                     f"{iqr(V['random(E)'])} | {iqr(V['val'])} | {vr} (p={pv:.3f}) | {bv} | {br} | {note} |")
            if res["val"]: beats_val.append((exp,cond,res["val"]))
            if res["random(E)"]: beats_rand.append((exp,cond,res["random(E)"]))
            if not res["random(E)"]: nothing.append((exp,cond))
    phase1b(L)
    valfree(L)
    signals_and_degeneracy(L)
    metric_robustness(L)
    L.append("\n## 17. Verdict across every cell\n")
    n_proxy_wins = len([x for x in beats_val if any(str(t).startswith("fg_") or str(t).startswith("val+") for t in x[2])])
    L.append(f"* A proxy Holm-beats `val` in **{n_proxy_wins}** "
             f"of **{sum(len(e) for _,_,e in POOLS)}** cells.")
    L.append("* Cells where **nothing at all** beats choosing blind: "
             + (", ".join(f"`{e}/{c}`" for e,c in nothing) or "none"))
    L.append("* Those counts cover the **shared-pool** families only, where every "
             "selector ranks one fixed set of 48 models. The sequential families are "
             "counted separately because a method there runs its own search: "
             "proxy-steered search Holm-beats `tpe_val` on **1** of 8 non-degenerate "
             "Phase 1B conditions (`lipo_scaffold`, and it is the study's only "
             "metric-robust proxy win — section 16.1) and is Holm-worse on 2 "
             "(`cond7`, `gdsc_drug`); with validation deleted entirely, the signal "
             "Holm-beats train-MSE alone on **6 of 6** arms on `cond7` and **0 of 6** "
             "on `hydro` (section 12).")
    L.append("")
    L.append("| cell | selectors that Holm-beat `val` |")
    L.append("|---|---|")
    for e,c,sl in beats_val:
        marked = ", ".join(("**`random(E)`** (i.e. `val` is below chance here)" if x=="random(E)"
                            else "`"+x+"`") for x in sl)
        L.append(f"| `{e}/{c}` | {marked} |")
    coverage(L)
    with open(OUT,"w") as f: f.write("\n".join(L)+"\n")
    print(f"wrote {OUT}")

if __name__=="__main__":
    main()
