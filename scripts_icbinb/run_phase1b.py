#!/usr/bin/env python3
"""Phase 1B — sequential HPO (plan §10). 9 methods x {cond7, hydro} x 10 reps x 48 trials.

Trials within a run are inherently sequential (the optimizer needs each result before
proposing the next), so parallelism is ACROSS the 180 (method, condition, rep) jobs.
Resumable per job; incumbent trajectories give budget curves for free (§10.5).
"""
from __future__ import annotations
import argparse, json, os, sys, time
import multiprocessing as mp
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import numpy as np
import torch

OUT = ROOT / "artifacts/results" / "phase1b_sequential"
CONDITIONS = ["cond7", "hydro"]        # ascending cost
REPS, BUDGET = 10, 48
BUDGET_MARKS = [8, 16, 24, 48]

def shard(method, cond, run): return OUT / "runs" / method / cond / f"outer{run:03d}.parquet"


def ckpt(method, cond, run):
    """Mid-shard checkpoint. A 1B shard is 48 SEQUENTIAL trials -- the sampler
    proposes trial t from trials 1..t-1 -- so a shard cannot simply be restarted
    part-way without changing the search. We therefore snapshot, after every trial:

      * the rows completed so far,
      * the pickled optimiser, and
      * the numpy AND torch global RNG states.

    All three are required. Verified in isolation: replaying the observation history
    alone reproduces TPE exactly but NOT HEBO's GeneralBO, which consumes the global
    RNG -- so a replay-based resume would silently produce a different trajectory for
    the three `hebo_*_mo` arms. Pickle + both RNG states is exact for all 9 methods.
    """
    return OUT / "runs" / method / cond / f"outer{run:03d}.ckpt.pkl"


def _out_root(): return str(OUT)

def deploy_index(rows, base_col, sig_col=None):
    """Plan §10.3: within-budget rank-sum over observed trials (rank(base)+rank(signal));
    falls back to argmin(base) when no signal is given."""
    from scipy.stats import rankdata
    ok = [i for i, r in enumerate(rows) if r["status_train"] == "ok" and np.isfinite(r[base_col])]
    if not ok: return None
    if sig_col is None:
        return min(ok, key=lambda i: rows[i][base_col])
    ok2 = [i for i in ok if np.isfinite(rows[i].get(sig_col, np.nan))]
    if not ok2: return min(ok, key=lambda i: rows[i][base_col])
    s = rankdata([rows[i][base_col] for i in ok2]) + rankdata([rows[i][sig_col] for i in ok2])
    return ok2[int(np.argmin(s))]

def _job(args):
    # `out_root` MUST travel in the args tuple: _job runs in a `spawn` worker, which
    # re-imports this module fresh, so the parent's _set_out() never reaches it.
    # Building the path from the module-level OUT here silently wrote exploratory
    # runs into the pre-registered tree.
    method, cond, run, phash, out_root = args
    _set_out(Path(out_root))
    from icbinb.datasets import load
    from icbinb.seeds import seeds_for
    from icbinb.train import train_candidate
    from icbinb.sequential import make
    dst = shard(method, cond, run)
    if dst.exists(): return (method, cond, run, 0, 0.0)
    sd = seeds_for(run)
    sp = load(cond, sd.split_seed)
    base = "val_mse_std"

    # ---- resume from a mid-shard checkpoint if one exists ----
    import pickle
    cp = ckpt(method, cond, run)
    rows, t0_trial = [], 0
    opt = sig = None
    if cp.exists():
        try:
            with open(cp, "rb") as fh:
                st = pickle.load(fh)
            opt, sig = st["opt"], st["sig"]
            rows, t0_trial = st["rows"], st["trial"]
            np.random.set_state(st["np_rng"]); torch.set_rng_state(st["torch_rng"])
            print(f"  resume {method}/{cond}/outer{run:03d} at trial {t0_trial}/{BUDGET}",
                  flush=True)
        except Exception as e:
            print(f"  checkpoint unreadable ({type(e).__name__}), restarting shard",
                  flush=True)
            rows, t0_trial, opt = [], 0, None
    if opt is None:
        opt, sig = make(method, sd.sampler_seed)
    t0 = time.time()          # NB: do NOT reset `rows` -- it may hold restored trials
    for t in range(t0_trial, BUDGET):
        hp = opt.suggest()
        r = train_candidate(sp, hp, sd, t, worker_id=os.getpid() % 1000,
                            extra=dict(experiment="phase1b", arm="sequential", method=method,
                                       dataset=cond.split("_")[0], condition=cond,
                                       outer_run=run, trial_index=t, protocol_hash=phash))
        r.pop("test_preds", None)
        rows.append(r)
        # feed the optimizer; diverged trials get a large finite penalty so BO keeps working
        bad = (r["status_train"] != "ok")
        yv = 1e6 if bad or not np.isfinite(r.get(base, np.nan)) else r[base]
        if sig is None: opt.observe(hp, yv)
        else:
            ys = r.get(sig, np.nan)
            opt.observe(hp, [yv, 1e6 if (bad or not np.isfinite(ys)) else ys])
        # ---- checkpoint after every trial (atomic) ----
        cp.parent.mkdir(parents=True, exist_ok=True)
        ctmp = cp.with_suffix(".tmp")
        with open(ctmp, "wb") as fh:
            pickle.dump({"opt": opt, "sig": sig, "rows": rows, "trial": t + 1,
                         "np_rng": np.random.get_state(),
                         "torch_rng": torch.get_rng_state()}, fh)
        os.replace(ctmp, cp)
    # incumbent trajectory: deploy rule applied to the first t trials
    traj = {}
    for t in range(1, BUDGET + 1):
        i = deploy_index(rows[:t], base, sig)
        traj[t] = None if i is None else float(rows[i]["test_mse_std"])
    import pandas as pd
    df = pd.DataFrame(rows)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp"); df.to_parquet(tmp, index=False); os.replace(tmp, dst)
    if cp.exists():
        try: cp.unlink()          # shard complete -> checkpoint no longer needed
        except OSError: pass
    tdir = OUT / "trajectories" / method / cond; tdir.mkdir(parents=True, exist_ok=True)
    ttmp = tdir / f"outer{run:03d}.json.tmp"
    ttmp.write_text(json.dumps({"method":method,"condition":cond,"outer_run":run,
                                "trajectory":traj,
                                "budget_marks":{b:traj[b] for b in BUDGET_MARKS}}, indent=1))
    os.replace(ttmp, tdir / f"outer{run:03d}.json")
    return (method, cond, run, len(df), time.time() - t0)

def _set_out(p):
    global OUT
    OUT = p


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS)
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None,
                    help="override output root (exploratory runs go elsewhere so they "
                         "can never be pooled with the pre-registered tree)")
    a = ap.parse_args()
    if a.out:
        _set_out(Path(a.out)); OUT.mkdir(parents=True, exist_ok=True)
    from icbinb.sequential import METHODS
    methods = a.methods or METHODS
    phash = (ROOT/"artifacts/results"/"protocol"/"protocol_hash.txt").read_text().strip()
    jobs = [(m, c, r, phash, _out_root()) for c in a.conditions for r in range(1, a.reps+1) for m in methods]
    todo = [j for j in jobs if not shard(j[0], j[1], j[2]).exists()]
    print(f"Phase 1B: {len(jobs)} jobs ({len(todo)} to run) x {BUDGET} trials, workers={a.workers}", flush=True)
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        for m, c, r, n, dt in pool.imap_unordered(_job, todo):
            if n: print(f"  {m:<16} {c:<7} outer{r:03d}  {n} trials  {dt:6.1f}s", flush=True)
    print(f"done in {(time.time()-t0)/60:.1f} min")
