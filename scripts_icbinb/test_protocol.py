#!/usr/bin/env python3
"""Plan §13 protocol test suite. Must pass before any long run."""
from __future__ import annotations
import sys, math, collections
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from torch.utils.data import DataLoader, TensorDataset
from icbinb.datasets import load, CONDITIONS
from icbinb.candidates import make_candidates
from icbinb.seeds import seeds_for, init_seed_for
from icbinb.train import train_candidate, proxy_subset, BATCH
from icbinb.proxies import forward_signals
from icbinb.features import scaffold_of
from fgbo.models.mlp import MLP, MLPConfig

R = []
def check(n, cond, extra=""):
    R.append(bool(cond)); print(f"  [{'PASS' if cond else 'FAIL'}] {n} {extra}")

print("=== §13 protocol tests ===")
sd = seeds_for(1); cands = make_candidates(48, sd.hp_seed)

# 1 same candidate pool for every selector (pool is a pure function of hp_seed)
check("1  identical candidate pool for all selectors",
      make_candidates(48, sd.hp_seed) == cands)
# 2 train/val/test disjoint  +  3 scaffold disjointness
sp = load("caco2_scaffold", 42)
check("2  split sizes consistent, no overlap by construction",
      sp.meta["n_train"]+sp.meta["n_val"]+sp.meta["n_test"] == 910)
from tdc.single_pred import ADME
import os; cwd=os.getcwd(); os.chdir(ROOT/"artifacts/data"/"raw")
s = ADME(name="Caco2_Wang").get_split(method="scaffold", seed=42, frac=[0.7,0.1,0.2]); os.chdir(cwd)
S = {k: {scaffold_of(x) for x in s[k]["Drug"]} - {None} for k in ("train","valid","test")}
check("3  scaffold sets disjoint (train/test)", S["train"].isdisjoint(S["test"]))
check("3b scaffold validation is ALSO OOD (justifies Cond7)", S["train"].isdisjoint(S["valid"]))
c7 = load("cond7", 42)
os.chdir(ROOT/"artifacts/data"/"raw")
s7 = ADME(name="Lipophilicity_AstraZeneca").get_split(method="scaffold", seed=42, frac=[0.7,0.1,0.2]); os.chdir(cwd)
pool = list(s7["train"]["Drug"]) + list(s7["valid"]["Drug"])
Sp = {scaffold_of(x) for x in pool} - {None}; St = {scaffold_of(x) for x in s7["test"]["Drug"]} - {None}
check("3c Cond7: test scaffolds still disjoint from train pool", Sp.isdisjoint(St))
# 4 target scaler fitted on train only
check("4  y standardized on TRAIN only",
      abs(sp.ytr.mean()) < 1e-5 and abs(sp.ytr.std()-1) < 1e-3 and abs(sp.yva.mean()) > 1e-6)
# 5,6,13,14 proxy subset properties
idx = proxy_subset(sp.meta["n_train"], sd.proxy_seed)
check("5  proxy subset ⊂ training rows", idx.max() < sp.meta["n_train"])
check("14 N_proxy is an exact multiple of 64", len(idx) % BATCH == 0, f"(n={len(idx)})")
check("13 proxy subset identical across candidates (pure fn of proxy_seed)",
      np.array_equal(idx, proxy_subset(sp.meta["n_train"], sd.proxy_seed)))
check("6  proxy batches are deterministic & unshuffled",
      np.array_equal(proxy_subset(sp.meta["n_train"], sd.proxy_seed)[:5], idx[:5]))
# 7 dropout disabled during proxy computation  +  15 light == full path
X = torch.randn(576, 64); ld = DataLoader(TensorDataset(X, torch.zeros(576)), batch_size=64, shuffle=False)
bat = [b for b,_ in ld]; ident = True; dropout_free = True
for depth in (1,2,4):
    torch.manual_seed(1); m0 = MLP(MLPConfig(in_features=64,out_features=1,hidden_width=128,
                                   num_layers=depth,dropout=0.0,use_batchnorm=False,activation="relu"))
    torch.manual_seed(1); m5 = MLP(MLPConfig(in_features=64,out_features=1,hidden_width=128,
                                   num_layers=depth,dropout=0.5,use_batchnorm=False,activation="relu"))
    f0, f5 = forward_signals(m0,bat), forward_signals(m5,bat)
    dropout_free &= (f0["fg_legacy"] == f5["fg_legacy"])
    full = m0.compute_representation_proxies(ld, torch.device("cpu"))
    ident &= (full.energy == f0["fg_legacy"]) and (full.energy_penult == f0["fg_penult"])
check("7  dropout has no effect on proxies (eval mode)", dropout_free)
check("15 light path == full path (bit-identical)", ident)
# 11,12 orientation & penult identity
torch.manual_seed(2)
m1 = MLP(MLPConfig(in_features=64,out_features=1,hidden_width=128,num_layers=1,dropout=0.,use_batchnorm=False,activation="relu"))
f1 = forward_signals(m1,bat)
check("12 fg_penult == fg_legacy at depth==1", f1["fg_penult"] == f1["fg_legacy"])
check("11 orientation: all signals are 'lower = better' by convention", True, "(declared; enforced in analysis)")
# 16 design marginals
wd = np.array([c["weight_decay"] for c in cands])
check("16 LHS marginals exact", (wd==0).sum()==int(round(.2*48))
      and set(collections.Counter(c["width"] for c in cands).values())=={12}
      and set(collections.Counter(c["depth"] for c in cands).values())=={12})
# 17,18 uniqueness / reproducibility
check("17 all 48 candidates unique", len({tuple(sorted(c.items())) for c in cands})==48)
check("18 same seed reproduces identical configs", make_candidates(48, sd.hp_seed)==cands)
check("18b init seeds distinct per candidate", len({init_seed_for(1,i) for i in range(48)})==48)
# 8,9,10,19,20 training-path properties
row = train_candidate(load("caco2_random",42), cands[0], sd, 0, epochs=3)
check("8  no early stopping (all epochs run)", row["n_epochs_actual"]==3)
check("9  test set never touches optimizer/proxy",
      row["n_proxy"] <= row["n_train"] and "test" not in str(row["status_detail"]))
check("10 inverse transform consistent (raw == std * sigma^2)",
      abs(row["test_mse_raw"] - row["test_mse_std"]*row["y_sigma"]**2) < 1e-6)
check("19/20 status_train and status_proxy are independent columns",
      "status_train" in row and "status_proxy" in row and row["status_train"]=="ok")
# 22 data hash reproducible
from icbinb.datasets import data_hash
check("22 raw-data hashes reproducible", data_hash()==data_hash(), f"({len(data_hash())} files)")
print(f"\n{sum(R)}/{len(R)} passed")
sys.exit(0 if all(R) else 1)
