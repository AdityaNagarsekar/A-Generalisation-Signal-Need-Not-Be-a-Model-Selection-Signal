#!/usr/bin/env python3
"""Plan §4 pilot gate: is 100 epochs enough? Inspects ONLY train/val curves and
optimisation stability — never test. Run before the protocol lock."""
from __future__ import annotations
import sys, json, multiprocessing as mp
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

CONDS = ["caco2_random","caco2_scaffold","lipo_random","lipo_scaffold","cond7","amylase","hydro"]
NCFG = 10

def job(args):
    cond, cid = args
    from icbinb.datasets import load
    from icbinb.candidates import make_candidates
    from icbinb.seeds import seeds_for
    from icbinb.train import train_candidate
    sd = seeds_for(1)
    sp = load(cond, sd.split_seed % 100 if cond.startswith(("lipo","caco2","cond7")) else 42)
    hp = make_candidates(48, sd.hp_seed)[cid]
    r = train_candidate(sp, hp, sd, cid, epochs=100)
    # ONLY train/val quantities are returned — test is never surfaced (§9 leakage rule)
    return dict(cond=cond, cid=cid, status=r["status_train"],
                tr25=r.get("train_mse_e25"), tr50=r.get("train_mse_e50"),
                tr100=r.get("train_mse_std"), va25=r.get("val_mse_e25"),
                va50=r.get("val_mse_e50"), va100=r.get("val_mse_std"),
                secs=r["runtime_train_sec"], depth=hp["depth"], width=hp["width"])

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    jobs = [(c,i) for c in CONDS for i in range(NCFG)]
    with mp.Pool(8) as pool: rows = pool.map(job, jobs)
    Path(ROOT/"artifacts/results"/"diagnostics").mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(ROOT/"artifacts/results"/"diagnostics"/"pilot_epoch_gate.json","w"), indent=1)
    print(f"{'condition':<16}{'ok':>4}{'div':>4}{'med s':>7} | "
          f"{'train 25→50':>13}{'train 50→100':>14} | {'val 50→100':>12}  verdict")
    print("-"*92)
    import statistics as st
    for c in CONDS:
        rs=[r for r in rows if r["cond"]==c]; ok=[r for r in rs if r["status"]=="ok"]
        if not ok: print(f"{c:<16}   0{len(rs):>4}  — all diverged"); continue
        d1=st.median([(r['tr25']-r['tr50'])/max(r['tr25'],1e-12) for r in ok])
        d2=st.median([(r['tr50']-r['tr100'])/max(r['tr50'],1e-12) for r in ok])
        v2=st.median([(r['va50']-r['va100'])/max(abs(r['va50']),1e-12) for r in ok])
        verdict = "CONVERGED (keep 100)" if d2 < 0.10 else "STILL IMPROVING -> raise epochs"
        print(f"{c:<16}{len(ok):>4}{len(rs)-len(ok):>4}{st.median([r['secs'] for r in ok]):>7.1f} | "
              f"{d1:>12.1%}{d2:>13.1%} | {v2:>11.1%}  {verdict}")
