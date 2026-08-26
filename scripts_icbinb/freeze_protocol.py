#!/usr/bin/env python3
"""Write the protocol lock (plan §9): protocol.yaml -> canonical JSON -> SHA-256,
plus env.json and data_hashes.json. Every result row carries protocol_hash+data_hash.

Run ONCE before the first real run, and git-commit the output.
"""
from __future__ import annotations
import hashlib, json, platform, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
OUT = ROOT / "artifacts/results" / "protocol"

def protocol() -> dict:
    from icbinb.candidates import (LR_LO, LR_HI, WD_LO, WD_HI, WD_ZERO_FRAC,
                                   DROPOUT_LO, DROPOUT_HI, WIDTHS, DEPTHS)
    from icbinb.train import EPOCHS, BATCH, PROXY_CAP, DIAG_EPOCHS
    from icbinb.features import MORGAN_RADIUS, MORGAN_BITS, FLIP_VOCAB
    return {
        "study": "ICBINB-BIO 2026 — model selection under biological shift",
        "conditions": ["lipo_random","lipo_scaffold","caco2_random","caco2_scaffold",
                       "cond7","amylase","hydro"],
        "phase1a_reps": {"lipo_random":30,"lipo_scaffold":30,"caco2_random":20,
                         "caco2_scaffold":20,"amylase":20,"hydro":30,"cond7":20},
        "phase1b": {"conditions":["cond7","hydro"], "reps":10, "budget":48,
                    "methods":["random_val","tpe_val","hebo_val","tpe_fg_mo","hebo_fg_mo",
                               "tpe_hess_mo","hebo_hess_mo"],
                    "primary_comparison":"tpe_fg_mo vs tpe_val"},
        "n_candidates": 48, "design": "scrambled Latin hypercube",
        "search_space": {"lr":[LR_LO,LR_HI],"weight_decay":[WD_LO,WD_HI],
                         "wd_zero_frac":WD_ZERO_FRAC,"dropout":[DROPOUT_LO,DROPOUT_HI],
                         "width":list(WIDTHS),"depth":list(DEPTHS)},
        "training": {"optimizer":"Adam(coupled L2 weight_decay)","batch_size":BATCH,
                     "epochs":EPOCHS,"activation":"relu","scheduler":None,"loss":"mse",
                     "precision":"fp32","early_stopping":False,"diag_epochs":list(DIAG_EPOCHS)},
        "proxy": {"batch_size":BATCH,"cap":PROXY_CAP,"multiple_of_batch":True,
                  "eval_mode":True,"shuffle":False},
        "signals": {"primary":"fg_legacy","secondary":["fg_penult","hess_top"],
                    "hessian":{"method":"power_iteration_hvp","n_iter":30,"tol":1e-3}},
        "features": {"morgan_radius":MORGAN_RADIUS,"morgan_bits":MORGAN_BITS,
                     "protein_vocab":FLIP_VOCAB,"protein_pad":"right_zero_allzero_rows"},
        "primary_metric": "paired delta OOD loss",
        "seeds": "base=10000*outer_run; +1 split, +2 hp, +3 init, +4 loader, +5 proxy, +6 sampler",
    }

def env() -> dict:
    import torch, numpy, scipy, pandas, sklearn, rdkit, optuna, pyarrow
    import tdc, hebo, statsmodels
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "machine": platform.machine(), "torch": torch.__version__,
            "numpy": numpy.__version__, "scipy": scipy.__version__,
            "pandas": pandas.__version__, "sklearn": sklearn.__version__,
            "rdkit": rdkit.__version__, "optuna": optuna.__version__,
            "hebo": hebo.__version__, "pyarrow": pyarrow.__version__,
            "PyTDC": getattr(tdc, "__version__", "1.1.15"),
            "statsmodels": statsmodels.__version__,
            "git_sha": subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,
                                      capture_output=True,text=True).stdout.strip()}

if __name__ == "__main__":
    from icbinb.datasets import data_hash
    OUT.mkdir(parents=True, exist_ok=True)
    p = protocol()
    blob = json.dumps(p, sort_keys=True).encode()
    h = hashlib.sha256(blob).hexdigest()
    (OUT/"protocol.json").write_text(json.dumps(p, indent=2, sort_keys=True))
    (OUT/"protocol_hash.txt").write_text(h + "\n")
    (OUT/"env.json").write_text(json.dumps(env(), indent=2, sort_keys=True))
    dh = data_hash()
    (OUT/"data_hashes.json").write_text(json.dumps(dh, indent=2, sort_keys=True))
    print(f"protocol_hash = {h}")
    print(f"data files hashed: {len(dh)}")
    for k,v in env().items(): print(f"  {k:<12} {v}")
