#!/usr/bin/env python3
"""Phase 1A — shared candidate pool (plan §5).

Trains 48 LHS candidates per outer run, writes ONE parquet row per candidate
immediately (atomic tmp->rename) plus an npz of val/test predictions per outer run.
Resumable: completed (condition, outer_run) shards are skipped.

  python scripts_icbinb/run_phase1a.py                    # all conditions, cheap first
  python scripts_icbinb/run_phase1a.py --conditions caco2_random --reps 2
"""
from __future__ import annotations
import argparse, json, os, sys, time
import multiprocessing as mp
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import numpy as np

OUT = ROOT / "artifacts/results" / "phase1a_pool"
LOGS = ROOT / "artifacts/results" / "logs"
# ascending cost order (plan §15.2) — cheap conditions first so bugs surface fast
ORDER = ["caco2_random","caco2_scaffold","lipo_random","lipo_scaffold","cond7","amylase","hydro"]
REPS  = {"caco2_random":20,"caco2_scaffold":20,"lipo_random":30,"lipo_scaffold":30,
         "cond7":20,"amylase":20,"hydro":30}
NCAND = 48

def shard(cond: str, run: int) -> Path:
    return OUT / "candidates" / cond / f"outer{run:03d}.parquet"

def _one(args):
    cond, run, cid, phash = args
    from icbinb.datasets import load
    from icbinb.candidates import make_candidates
    from icbinb.seeds import seeds_for
    from icbinb.train import train_candidate
    sd = seeds_for(run)
    sp = load(cond, sd.split_seed)
    hp = make_candidates(NCAND, sd.hp_seed)[cid]
    r = train_candidate(sp, hp, sd, cid, worker_id=os.getpid() % 1000,
                        extra=dict(experiment="phase1a", arm="pool", method="lhs_pool",
                                   dataset=cond.split("_")[0], condition=cond,
                                   outer_run=run, trial_index=cid,
                                   protocol_hash=phash))
    return r

def run_shard(cond: str, run: int, phash: str, workers: int) -> int:
    dst = shard(cond, run)
    if dst.exists(): return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with mp.Pool(workers) as pool:
        rows = pool.map(_one, [(cond, run, c, phash) for c in range(NCAND)])
    preds = np.stack([r.pop("test_preds") for r in rows])
    pdir = OUT / "predictions" / cond; pdir.mkdir(parents=True, exist_ok=True)
    # predictions first, parquet last: the PARQUET is the completion marker, so an
    # interrupt between the two leaves a stale npz that the re-run simply overwrites.
    # NOTE: numpy APPENDS .npz when the filename does not already end in .npz, so the
    # temp name must itself end in .npz or os.replace() will not find it.
    ptmp = pdir / f"outer{run:03d}.tmp.npz"
    np.savez_compressed(ptmp, test_preds=preds)
    os.replace(ptmp, pdir / f"outer{run:03d}.npz")
    import pandas as pd
    df = pd.DataFrame(rows)
    tmp = dst.with_suffix(".tmp"); df.to_parquet(tmp, index=False); os.replace(tmp, dst)
    div = int((df.status_train != "ok").sum())
    print(f"  {cond:<16} outer{run:03d}  {len(df):>3} rows  {time.time()-t0:6.1f}s  "
          f"diverged={div}  median_train={df.train_mse_std.median():.4f}", flush=True)
    return len(df)

def _set_out(p):
    global OUT
    OUT = p


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=ORDER)
    ap.add_argument("--reps", type=int, default=None, help="override rep count")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None,
                    help="override output root (exploratory runs go elsewhere so they "
                         "can never be pooled with the pre-registered tree)")
    a = ap.parse_args()
    if a.out:
        _set_out(Path(a.out)); OUT.mkdir(parents=True, exist_ok=True)
    phash = (ROOT/"artifacts/results"/"protocol"/"protocol_hash.txt").read_text().strip()
    print(f"protocol_hash={phash[:16]}...  workers={a.workers}")
    total = 0; t0 = time.time()
    for cond in a.conditions:
        n = a.reps if a.reps is not None else REPS[cond]
        print(f"\n=== {cond} ({n} outer runs x {NCAND} candidates) ===", flush=True)
        for run in range(1, n + 1):
            total += run_shard(cond, run, phash, a.workers)
    print(f"\ndone: {total} trainings in {(time.time()-t0)/60:.1f} min")
