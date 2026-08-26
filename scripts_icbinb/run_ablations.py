#!/usr/bin/env python3
"""Ablations (plan §8): fixed-architecture, val-free pool, val-free sequential.

  fixed_arch   d_in->256->256->1, search only (lr, wd, dropout); cond7 + hydro, 10 reps
  valfree_pool 48-candidate pool trained on train u val; 4 OOD conditions, 10 reps
  valfree_seq  6 MO methods on (train_mse, signal); cond7 + hydro, 10 reps

All resumable per shard.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import multiprocessing as mp
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import numpy as np

ABL = ROOT / "artifacts/results" / "ablations"
NCAND, REPS = 48, 10
FIXED_CONDS = ["cond7", "hydro"]
VF_POOL_CONDS = ["caco2_scaffold", "lipo_scaffold", "amylase", "hydro"]
VF_METHODS = ["tpe_fg_mo","hebo_fg_mo","tpe_penult_mo","hebo_penult_mo","tpe_hess_mo","hebo_hess_mo"]

def _fixed_arch_hps(hp_seed, n=NCAND):
    """Architecture pinned; only (lr, wd, dropout) vary — same LHS machinery."""
    from icbinb.candidates import make_candidates
    out = []
    for c in make_candidates(n, hp_seed):
        out.append({**c, "width": 256, "depth": 2})
    return out

def _pool_job(args):
    kind, cond, run, cid, phash = args
    from icbinb.datasets import load, fold_val_into_train
    from icbinb.candidates import make_candidates
    from icbinb.seeds import seeds_for
    from icbinb.train import train_candidate
    sd = seeds_for(run); sp = load(cond, sd.split_seed)
    if kind == "fixed_arch":
        hp = _fixed_arch_hps(sd.hp_seed)[cid]
    else:
        hp = make_candidates(NCAND, sd.hp_seed)[cid]; sp = fold_val_into_train(sp)
    return train_candidate(sp, hp, sd, cid, worker_id=os.getpid() % 1000,
                           extra=dict(experiment=kind, arm=kind, method=kind,
                                      dataset=cond.split("_")[0], condition=cond,
                                      outer_run=run, trial_index=cid, protocol_hash=phash))

def run_pool(kind, cond, run, phash, workers):
    dst = ABL / kind / cond / f"outer{run:03d}.parquet"
    if dst.exists(): return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with mp.Pool(workers) as pool:
        rows = pool.map(_pool_job, [(kind, cond, run, c, phash) for c in range(NCAND)])
    for r in rows: r.pop("test_preds", None)
    import pandas as pd
    tmp = dst.with_suffix(".tmp"); pd.DataFrame(rows).to_parquet(tmp, index=False); os.replace(tmp, dst)
    print(f"  {kind:<13} {cond:<15} outer{run:03d}  {len(rows)} rows  {time.time()-t0:6.1f}s", flush=True)
    return len(rows)

def _vfseq_job(args):
    method, cond, run, phash = args
    from icbinb.datasets import load, fold_val_into_train
    from icbinb.seeds import seeds_for
    from icbinb.train import train_candidate
    from icbinb.sequential import make
    import importlib.util
    spec = importlib.util.spec_from_file_location("p1b", ROOT/"scripts_icbinb"/"run_phase1b.py")
    p1b = importlib.util.module_from_spec(spec); spec.loader.exec_module(p1b)
    dst = ABL / "valfree_sequential" / method / cond / f"outer{run:03d}.parquet"
    if dst.exists(): return (method, cond, run, 0, 0.0)
    sd = seeds_for(run); sp = fold_val_into_train(load(cond, sd.split_seed))
    opt, sig = make(method, sd.sampler_seed)
    base = "train_mse_std"                      # val-free: train stands in for val
    rows, t0 = [], time.time()
    for t in range(p1b.BUDGET):
        hp = opt.suggest()
        r = train_candidate(sp, hp, sd, t, worker_id=os.getpid() % 1000,
                            extra=dict(experiment="valfree_sequential", arm="valfree_sequential",
                                       method=method, dataset=cond.split("_")[0], condition=cond,
                                       outer_run=run, trial_index=t, protocol_hash=phash))
        r.pop("test_preds", None); rows.append(r)
        bad = r["status_train"] != "ok"
        yv = 1e6 if bad or not np.isfinite(r.get(base, np.nan)) else r[base]
        ys = r.get(sig, np.nan)
        opt.observe(hp, [yv, 1e6 if (bad or not np.isfinite(ys)) else ys])
    traj = {t: (None if (i:=p1b.deploy_index(rows[:t], base, sig)) is None
                else float(rows[i]["test_mse_std"])) for t in range(1, p1b.BUDGET+1)}
    import pandas as pd
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp"); pd.DataFrame(rows).to_parquet(tmp, index=False); os.replace(tmp, dst)
    tdir = ABL / "valfree_sequential" / "_trajectories" / method / cond
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / f"outer{run:03d}.json").write_text(json.dumps(traj, indent=1))
    return (method, cond, run, len(rows), time.time()-t0)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+",
                    default=["fixed_arch","valfree_pool","valfree_sequential"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--reps", type=int, default=REPS)
    a = ap.parse_args()
    phash = (ROOT/"artifacts/results"/"protocol"/"protocol_hash.txt").read_text().strip()
    if "fixed_arch" in a.only:
        print("=== ablation: fixed architecture (d_in->256->256->1) ===", flush=True)
        for cond in FIXED_CONDS:
            for run in range(1, a.reps+1): run_pool("fixed_arch", cond, run, phash, a.workers)
    if "valfree_pool" in a.only:
        print("=== ablation: val-free pool (train u val) ===", flush=True)
        for cond in VF_POOL_CONDS:
            for run in range(1, a.reps+1): run_pool("valfree_pool", cond, run, phash, a.workers)
    if "valfree_sequential" in a.only:
        print("=== ablation: val-free sequential (6 MO methods) ===", flush=True)
        jobs=[(m,c,r,phash) for c in FIXED_CONDS for r in range(1,a.reps+1) for m in VF_METHODS]
        todo=[j for j in jobs if not (ABL/"valfree_sequential"/j[0]/j[1]/f"outer{j[2]:03d}.parquet").exists()]
        print(f"  {len(todo)}/{len(jobs)} jobs to run", flush=True)
        with mp.Pool(a.workers) as pool:
            for m,c,r,n,dt in pool.imap_unordered(_vfseq_job, todo):
                if n: print(f"  valfree_seq {m:<16} {c:<7} outer{r:03d}  {n} trials {dt:6.1f}s", flush=True)
