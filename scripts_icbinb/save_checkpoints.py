#!/usr/bin/env python3
"""Save 5 model checkpoints per condition -> artifacts/results/checkpoints/.

The main runs deliberately store no weights (~233 GB at fp32 for 31,680 models).
This retrains a small deterministic sample so the weights can be inspected.

REPRODUCTION, NOT RE-ROLL. Every model is retrained with the *exact* seed quintuple
and candidate id from the original run, so it is the same model the study reported —
not a fresh draw. The script verifies this by comparing the reproduced metrics
against the stored parquet row and refuses to claim a match if they differ.

Candidates are chosen by a fixed RNG (seed 20260822 + a SHA-256 digest of the
condition name — not Python's `hash()`, which is randomised per process) so the
selection is itself reproducible, and they are drawn to span the pool rather than cluster: one each from
the best / worst / median of the validation ranking plus two at random.
"""
from __future__ import annotations

import argparse, glob, hashlib, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
OUT = ROOT / "artifacts/results" / "checkpoints"
PICK_SEED = 20260822
N_PER_COND = 5

SOURCES = {  # condition -> (pool root, outer run to reproduce)
    "caco2_random":   ("phase1a_pool/candidates", 1),
    "caco2_scaffold": ("phase1a_pool/candidates", 1),
    "lipo_random":    ("phase1a_pool/candidates", 1),
    "lipo_scaffold":  ("phase1a_pool/candidates", 1),
    "cond7":          ("phase1a_pool/candidates", 1),
    "amylase":        ("phase1a_pool/candidates", 1),
    "hydro":          ("phase1a_pool/candidates", 1),
    "gdsc_drug":      ("exploratory/gdsc/phase1a/candidates", 1),
}


def choose(df: pd.DataFrame, n: int, rng) -> list[int]:
    """Span the pool: best / median / worst by validation, plus random others."""
    ok = df[(df.status_train == "ok") & df.val_mse_std.notna()]
    order = ok.sort_values("val_mse_std")
    anchors = [int(order.iloc[0].candidate_id),
               int(order.iloc[len(order) // 2].candidate_id),
               int(order.iloc[-1].candidate_id)]
    rest = [c for c in ok.candidate_id.tolist() if c not in anchors]
    extra = rng.choice(rest, size=max(0, n - len(anchors)), replace=False).tolist()
    return anchors + [int(x) for x in extra]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=list(SOURCES))
    ap.add_argument("--n", type=int, default=N_PER_COND)
    a = ap.parse_args()

    from icbinb.datasets import load
    from icbinb.candidates import make_candidates
    from icbinb.seeds import seeds_for
    from icbinb.train import train_candidate
    import icbinb.train as T

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for cond in a.conditions:
        sub, run = SOURCES[cond]
        files = sorted(glob.glob(str(ROOT / "artifacts/results" / sub / cond / "*.parquet")))
        if not files:
            print(f"  {cond}: no pool on disk, skipped"); continue
        df = pd.read_parquet(files[run - 1])
        # stable across processes: Python's hash() is randomised by PYTHONHASHSEED
        cond_key = int(hashlib.sha256(cond.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(PICK_SEED + cond_key % 10_000)
        cids = choose(df, a.n, rng)

        sd = seeds_for(run)
        sp = load(cond, sd.split_seed)
        cands = make_candidates(48, sd.hp_seed)
        d = OUT / cond
        d.mkdir(parents=True, exist_ok=True)
        print(f"\n{cond}: reproducing candidates {cids} from outer{run:03d}")
        for cid in cids:
            hp = cands[cid]
            # rebuild the model exactly as train_candidate does, then train it
            row, model, snaps = _train_and_keep(sp, hp, sd, cid, train_candidate, T)
            stored = df[df.candidate_id == cid].iloc[0]
            same = {k: (float(row[k]), float(stored[k]))
                    for k in ("train_mse_std", "val_mse_std", "test_mse_std",
                              "fg_legacy", "fg_penult", "hess_top")
                    if k in row and pd.notna(stored[k])}
            worst = max(abs(v - s) for v, s in same.values())
            torch.save({"state_dict": model.state_dict(),
                        "state_dict_by_epoch": {int(k): v for k, v in snaps.items()},
                        "hp": hp, "cond": cond,
                        "candidate_id": int(cid), "outer_run": run,
                        "seeds": sd.asdict(), "d_in": int(sp.Xtr.shape[1]),
                        "y_mu": float(sp.y_mu), "y_sigma": float(sp.y_sigma),
                        "metrics": {k: float(row[k]) for k in row
                                    if isinstance(row[k], (int, float, np.floating))},
                        "reproduction_max_abs_err": float(worst)},
                       d / f"cand{cid:02d}_outer{run:03d}.pt")
            flag = "OK" if worst < 1e-6 else f"MISMATCH {worst:.2e}"
            print(f"   cand {cid:>2}  w={hp['width']:<4} d={hp['depth']}  "
                  f"lr={hp['lr']:.1e}  test={row['test_mse_std']:.4f}   "
                  f"epochs saved {sorted(snaps)}   repro {flag}")
            manifest.append(dict(condition=cond, candidate_id=int(cid), outer_run=run,
                                 file=f"{cond}/cand{cid:02d}_outer{run:03d}.pt",
                                 width=hp["width"], depth=hp["depth"], lr=hp["lr"],
                                 test_mse_std=float(row["test_mse_std"]),
                                 reproduction_max_abs_err=float(worst)))
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    n_ok = sum(1 for m in manifest if m["reproduction_max_abs_err"] < 1e-6)
    print(f"\n{len(manifest)} checkpoints written to {OUT}")
    print(f"exact reproductions: {n_ok}/{len(manifest)}")


def _train_and_keep(sp, hp, sd, cid, train_candidate, T):
    """train_candidate() discards the model; re-run its body but keep the weights.

    Also snapshots weights at the epochs the run already records metrics for
    (DIAG_EPOCHS = 25, 50) plus the final epoch, so the saved weights line up
    one-to-one with the stored `*_e25` / `*_e50` diagnostic columns.
    """
    import copy
    keep = {"snaps": {}}
    orig = T.MLP
    orig_mse = T._mse
    class _Capture(orig):
        def __init__(self, *args, **kw):
            super().__init__(*args, **kw); keep["model"] = self
    # _mse is called exactly once per diagnostic epoch, first thing -> use it as the hook
    seen = {"n": 0}
    def _mse_hook(model, X, y):
        # the first call at each diagnostic epoch snapshots the weights
        eps = list(T.DIAG_EPOCHS)
        idx = seen["n"] // 2          # two _mse calls per diagnostic epoch (train, val)
        if seen["n"] % 2 == 0 and idx < len(eps):
            keep["snaps"][eps[idx]] = copy.deepcopy(model.state_dict())
        seen["n"] += 1
        return orig_mse(model, X, y)
    T.MLP = _Capture
    T._mse = _mse_hook
    try:
        row = train_candidate(sp, hp, sd, cid, worker_id=0,
                              extra=dict(experiment="checkpoint", arm="checkpoint",
                                         method="repro", dataset=hp.get("dataset", ""),
                                         condition="", outer_run=0, trial_index=0,
                                         protocol_hash=""))
    finally:
        T.MLP = orig
        T._mse = orig_mse
    keep["snaps"][int(row.get("n_epochs_actual", 100))] = keep["model"].state_dict()
    return row, keep["model"], keep["snaps"]


if __name__ == "__main__":
    main()
