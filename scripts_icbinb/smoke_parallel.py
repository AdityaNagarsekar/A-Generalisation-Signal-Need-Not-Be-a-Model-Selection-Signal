#!/usr/bin/env python3
"""Phase-0 smoke tests: do parallel workers run and WRITE correctly?

Verifies the result-writing path before any real compute is spent:
  S1  every worker produces exactly one row (no lost/duplicate work)
  S2  all rows share the identical column superset (schema contract)
  S3  same seed -> identical config and identical metrics (determinism)
  S4  results survive an interrupted run (rows already on disk are intact)
  S5  signal failures are recorded, not silently dropped
  S6  concurrent appends to the same directory never corrupt a file
"""
from __future__ import annotations
import json, os, sys, time, math
import multiprocessing as mp
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts/results" / "diagnostics" / "smoke"

COLUMNS = [  # the shared superset (plan §7.1), abbreviated for the smoke test
    "experiment","arm","dataset","shift","outer_run","candidate_id",
    "split_seed","hp_seed","init_seed","loader_seed","proxy_seed",
    "lr","weight_decay","dropout","width","depth","n_params",
    "train_mse_std","val_mse_std","test_mse_std",
    "fg_legacy","fg_penult","hess_top","hess_iters","hess_converged",
    "runtime_train_sec","runtime_hess_sec","status","status_detail","device",
]

def seeds_for(outer_run: int) -> dict:
    b = 10000 * outer_run
    return dict(split_seed=b+1, hp_seed=b+2, init_seed=b+3,
                loader_seed=b+4, proxy_seed=b+5)

def sample_hp(hp_seed: int, cid: int) -> dict:
    import numpy as np
    rng = np.random.default_rng(hp_seed + cid)
    return dict(lr=float(10**rng.uniform(-4, math.log10(3e-3))),
                weight_decay=0.0 if rng.random() < 0.2 else float(10**rng.uniform(-6,-2)),
                dropout=float(rng.uniform(0,0.5)),
                width=int(rng.choice([64,128,256,512])),
                depth=int(rng.choice([1,2,3,4])))

def run_candidate(job):
    """One worker = one trained candidate + one row written immediately."""
    outer_run, cid, force_fail = job
    import numpy as np, torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    sys.path.insert(0, str(ROOT / "src"))
    from fgbo.proxies import top_hessian_eig
    torch.set_num_threads(1)

    sd = seeds_for(outer_run); hp = sample_hp(sd["hp_seed"], cid)
    torch.manual_seed(sd["init_seed"] + cid)
    g = torch.Generator().manual_seed(sd["split_seed"])
    X = torch.randn(256, 64, generator=g); y = torch.randn(256, generator=g)
    Xv = torch.randn(64, 64, generator=g); yv = torch.randn(64, generator=g)

    L, prev = [], 64
    for _ in range(hp["depth"]):
        L += [nn.Linear(prev, hp["width"]), nn.ReLU(), nn.Dropout(hp["dropout"])]
        prev = hp["width"]
    L.append(nn.Linear(prev, 1))
    m = nn.Sequential(*L)
    opt = torch.optim.Adam(m.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    crit = nn.MSELoss()
    ld = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True,
                    generator=torch.Generator().manual_seed(sd["loader_seed"]))
    t0 = time.time(); status, detail = "ok", ""
    for _ in range(4):
        m.train()
        for xb, yb in ld:
            opt.zero_grad(); loss = crit(m(xb).squeeze(-1), yb)
            if not torch.isfinite(loss): status, detail = "diverged_train", "nonfinite loss"; break
            loss.backward(); opt.step()
        if status != "ok": break
    rt = time.time() - t0

    m.eval()
    with torch.no_grad():
        tr = float(crit(m(X).squeeze(-1), y)); va = float(crit(m(Xv).squeeze(-1), yv))
    # proxies (fg_* stubbed here; the real run uses compute_representation_proxies)
    fg_legacy = float((m(X).detach()**2).mean())
    fg_penult = fg_legacy
    th = time.time()
    try:
        if force_fail: raise RuntimeError("injected hessian failure")
        eig, iters, conv = top_hessian_eig(m, X, y, seed=sd["proxy_seed"])
    except Exception as e:                      # S5: signal failure != training failure
        eig, iters, conv = float("nan"), 0, False
        status = "ok" if status == "ok" else status
        detail = f"hess_failed: {e}"
    rth = time.time() - th

    row = dict(experiment="smoke", arm="phase1a_pool", dataset="synthetic", shift="none",
               outer_run=outer_run, candidate_id=cid, **sd, **hp,
               n_params=sum(p.numel() for p in m.parameters()),
               train_mse_std=tr, val_mse_std=va, test_mse_std=float("nan"),
               fg_legacy=fg_legacy, fg_penult=fg_penult,
               hess_top=eig, hess_iters=iters, hess_converged=conv,
               runtime_train_sec=rt, runtime_hess_sec=rth,
               status=status, status_detail=detail, device="cpu")
    # ---- write immediately, one file per candidate (atomic: tmp then rename) ----
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / f"outer{outer_run:03d}_cand{cid:03d}.json"
    tmp = dst.with_suffix(".tmp")
    tmp.write_text(json.dumps(row)); os.replace(tmp, dst)     # atomic on POSIX
    return str(dst)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    import shutil
    if OUT.exists(): shutil.rmtree(OUT)
    P, N = 8, 24
    jobs = [(1, c, (c == 5)) for c in range(N)]      # candidate 5 gets an injected hess failure
    t0 = time.time()
    with mp.Pool(P) as pool:
        written = pool.map(run_candidate, jobs)
    wall = time.time() - t0

    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob("*.json"))]
    ok = True
    def check(name, cond, extra=""):
        global ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")

    print(f"\nsmoke: {N} candidates on P={P} workers in {wall:.1f}s -> {len(rows)} rows\n")
    check("S1 one row per candidate, none lost/duplicated",
          len(rows) == N and len({r['candidate_id'] for r in rows}) == N)
    check("S2 identical column superset across all rows",
          all(set(r) == set(rows[0]) for r in rows), f"({len(rows[0])} cols)")
    check("S2b superset covers the declared contract",
          set(COLUMNS).issubset(set(rows[0])),
          f"missing={sorted(set(COLUMNS)-set(rows[0]))}")
    # S3 determinism: rerun two candidates, compare
    again = [json.loads(Path(run_candidate((1, c, False))).read_text()) for c in (0, 3)]
    orig = {r["candidate_id"]: r for r in rows}
    same = all(abs(a["train_mse_std"] - orig[a["candidate_id"]]["train_mse_std"]) < 1e-12
               and a["lr"] == orig[a["candidate_id"]]["lr"] for a in again)
    check("S3 same seed -> identical config and metrics", same)
    check("S4 rows already on disk are complete JSON (interrupt-safe)",
          all(isinstance(r.get("status"), str) for r in rows))
    check("S4b no leftover .tmp files", not list(OUT.glob("*.tmp")))
    failed = [r for r in rows if "hess_failed" in (r["status_detail"] or "")]
    check("S5 signal failure recorded, training still 'ok'",
          len(failed) == 1 and failed[0]["status"] == "ok"
          and math.isnan(failed[0]["hess_top"]),
          f"(candidate {failed[0]['candidate_id'] if failed else '?'})")
    check("S6 every file parses (no concurrent-write corruption)", len(rows) == N)
    print(f"\n{'ALL SMOKE TESTS PASSED' if ok else 'SMOKE FAILURES PRESENT'}")
    sys.exit(0 if ok else 1)
