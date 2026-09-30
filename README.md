# A Generalisation Signal Need Not Be a Model-Selection Signal

Code, data and per-run results for the ICBINB-BIO @ NeurIPS 2026 paper (accepted,
poster).

The study asks a single question: **when a model pool is ranked by validation
loss, does that ranking survive a biological distribution shift - and can a
cheap geometric signal (activation energy, top Hessian eigenvalue, Hessian
trace) repair it when it does not?** The reported answer is largely negative,
and all results in the paper are reproducible. Individual runs are
committed as parquet, so the analysis and the figures can be regenerated
without retraining anything.

---

## 1. Layout

```
.
├── assets/                  project thumbnail + make_thumbnail.py
├── icbinb/                  study code: datasets, features, training, proxies,
│                            candidate pool, sequential HPO, analysis
├── src/fgbo/                MLP model + curvature proxies (installed package)
├── scripts_icbinb/          runnable entry points (see §5)
├── artifacts/
│   ├── data/                inputs
│   │   ├── raw/             benchmark source files + TDC molecule cache
│   │   └── features/        Morgan-fingerprint / one-hot caches (135 .npz)
│   └── results/             outputs — one row per trained model
│       ├── protocol/        FROZEN pre-registration (see §4)
│       ├── phase1a_pool/    fixed 48-candidate pool, all conditions
│       ├── phase1b_sequential/  sequential HPO, pre-registered conditions
│       ├── ablations/       fixed_arch, valfree_pool, valfree_sequential
│       ├── exploratory/     post-hoc only: gdsc, featshift, shift_severity,
│       │                    trace, phase1b_extra  (NEVER used to support a
│       │                    confirmatory claim — see §4)
│       ├── figures/         the 4 vector PDFs used in the paper
│       ├── diagnostics/     pilot epoch-gate + smoke tests
│       └── logs/            stdout from the long runs
├── requirements.txt         pinned to the environment the paper was run in
├── pyproject.toml
└── README.md                this file
```

---

## 2. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                 # exposes `fgbo` from src/
```

`requirements.txt` pins the exact versions recorded in
`artifacts/results/protocol/env.json`, which is the environment every reported
number was produced under (Python 3.10.18, torch 2.9.0, macOS/arm64).

**Run everything from the repository root.** The `icbinb` package is imported
from the working directory and all result paths are repo-root-relative. The
run scripts set `PYTHONPATH=.:src` themselves; if you invoke a module directly,
export it yourself:

```bash
export PYTHONPATH=.:src
```

---

## 3. Data — what ships, what downloads

| Source | Where it lives | How you get it |
|---|---|---|
| FLIP2 `amylase` (close→far) | `artifacts/data/raw/amylase/close_to_far.csv.gz` | **committed** |
| FLIP2 `hydro` (low→high) | `artifacts/data/raw/hydro/low_to_high.csv.gz` | **committed** |
| TDC Caco-2 (Wang) | `artifacts/data/raw/data/caco2_wang.tab` | **committed** (PyTDC re-downloads if absent) |
| TDC Lipophilicity (AstraZeneca) | `artifacts/data/raw/data/lipophilicity_astrazeneca.tab` | **committed** (PyTDC re-downloads if absent) |
| L1000 landmark map | `artifacts/data/raw/l1000/landmark_map.tsv` | **committed** |
| Morgan / one-hot feature caches | `artifacts/data/features/*.npz` | **committed** (see note below) |
| TDC GDSC2 (`DrugRes`) | `./data/gdsc2.pkl` at the repo root | **auto-downloads on first use** (~112 MB) |

The molecule loaders `chdir` into `artifacts/data/raw/` before calling PyTDC,
so their cache lands inside the committed tree. The GDSC2 loader does not, so
PyTDC writes it to `<cwd>/data/` — i.e. `./data/` when you run from the repo
root, which is gitignored. Nothing else is needed: run any GDSC script and
PyTDC fetches it.

The feature caches are committed on purpose. They are deterministic given a
fixed RDKit version, but committing them removes RDKit-version drift as a
source of irreproducibility. Delete `artifacts/data/features/` and they rebuild
from the raw files on the next run.

---

## 4. The frozen protocol

`artifacts/results/protocol/` is the pre-registration, written before the
confirmatory runs and never edited since:

| File | Contents |
|---|---|
| `protocol.json` | conditions, 48-candidate LHS search space, replication counts, Phase 1B methods, the declared primary comparison |
| `protocol_hash.txt` | `42554d911431f9de011072ee387d0a957c5e161f379e08ed650c8179463b42f1` |
| `data_hashes.json` | content hashes of the 4 raw benchmark files, keyed relative to `artifacts/data/raw/` |
| `env.json` | exact package versions + platform |

Every confirmatory run script re-reads `protocol_hash.txt` and stamps it into
its output, so a result file can always be traced to the protocol it ran under.
All 59,712 stored result rows carry the frozen hash (checked by
`scripts_icbinb/review_diagnostics.py`).

**Pre-registered confirmatory family:** `cond7`, `caco2_scaffold`, `amylase`,
`hydro`. `lipo_scaffold` was excluded in advance and is reported descriptively.
`lipo_scaffold` and `caco2_scaffold` are built by the same TDC scaffold split,
so the paper's Appendix B also reports the family with `lipo_scaffold` added,
as a sensitivity analysis.
Everything under `artifacts/results/exploratory/` is post-hoc by construction
and is never used to support a confirmatory claim.

**Seeds.** One *outer replication* `r` maps to a fixed six-seed tuple
`(10000r+1, …, 10000r+6)` for split / hyperparameters / init / loader / proxy /
sampler respectively (`icbinb/seeds.py`). The candidate pool is redrawn per
replication; the test set is redrawn too, except on FLIP2 where the benchmark's
own split is used verbatim.

Verify the whole protocol — 23 checks including raw-data hash reproducibility,
scaffold-set disjointness, train-only standardisation, and a bit-identical
light-path-vs-full-path comparison:

```bash
PYTHONPATH=.:src python3 scripts_icbinb/test_protocol.py
# -> 23/23 passed
```

---

## 5. Reproducing

### Everything, unattended

```bash
nohup ./scripts_icbinb/run_all.sh > artifacts/results/logs/run_all.log 2>&1 &
```

Runs Phase 1A → Phase 1B → the three ablations, in that order. Every stage is
resumable: re-running picks up exactly where it stopped, because completed
trials are checkpointed per trial. Worker count defaults to 5
(`WORKERS=n ./scripts_icbinb/run_all.sh` to change it) — this was tuned for an
11-core laptop; 8 workers pinned every core and destabilised the machine.

This is a multi-day run. The committed results let you skip it entirely.

### Individual stages

| Script | What it does |
|---|---|
| `run_phase1a.py` | trains the fixed 48-candidate pool per condition |
| `run_phase1b.py` | sequential HPO (random / TPE / HEBO × val / proxy-MO / hessian-MO), budget 48 |
| `run_ablations.py --only {fixed_arch,valfree_pool,valfree_sequential}` | the three ablations |
| `run_exploratory.sh`, `run_featshift.sh`, `run_trace.sh`, `run_phase1b_extra.sh` | post-hoc arms |
| `freeze_protocol.py` | rewrites the protocol lock — **do not run**, it would invalidate the pre-registration |

### Analysis and figures (fast — reads the committed parquet)

```bash
export PYTHONPATH=.:src
./scripts_icbinb/finish_section.sh      # regenerates every markdown analysis
python3 scripts_icbinb/make_figures.py  # -> artifacts/results/figures/{phase1a,phase1b}/
python3 scripts_icbinb/sweep_all.py     # master results sweep
python3 scripts_icbinb/make_metric_tables.py   # per-dataset tables
python3 scripts_icbinb/make_appendix_tex.py    # LaTeX appendix tables
python3 scripts_icbinb/review_diagnostics.py   # numbers added at camera-ready (see §6)
```

The four figure PDFs the paper uses are kept in `artifacts/results/figures/`
and in the paper's `figures/` folder. `make_appendix_tex.py` emits code
identifiers; the paper's appendix tables rename them (§6) and state whether they
use audited or unfiltered pools, so they were edited after generation.

The markdown reports land in `artifacts/results/summaries/`, which is
gitignored — they are rendered from the committed parquet, so they are outputs,
not sources. This README is the only markdown checked in.

---

## 6. Reading the paper against the code

Selector names in the paper map to code identifiers as follows (paper Appendix B,
Table 3):

| Paper | Code | Deploys the candidate with the lowest |
|---|---|---|
| Val | `val` | validation MSE |
| Proxy | `fg_legacy` | layer-averaged activation proxy (the pre-registered primary signal) |
| Proxy (pen.) | `fg_penult` | penultimate-layer activation proxy |
| λmax | `hess_top` | top Hessian eigenvalue of the training MSE, over all weights and biases |
| Val+Proxy, Val+Proxy (pen.), Val+λmax | `val+legacy`, `val+penult`, `val+hess` | rank sum of validation MSE and the signal |
| Train | `train_mse` | training MSE |
| #params | `n_params` | parameter count |
| Random | `random(E)` | none: expected deployment loss of a uniform pick |
| Oracle | `oracle` | deployment MSE (a bound, not a selector) |

Ties go to the lowest `candidate_id`, which is the Latin-hypercube draw order.
Figures label `cond7` as *Cond7* and `hydro` as *Hydro*.

**Audited pools.** The post-hoc degeneracy audit drops a candidate if
`pred_std_test < 1e-6` (a constant predictor) or `dead_relu_frac > 0.5` (more
than half the hidden ReLU units are zero on every example of the training proxy
subset). A replication with fewer than five surviving candidates would be
dropped; none is. Primary inference in the paper uses the unfiltered pools.

`scripts_icbinb/review_diagnostics.py` reproduces every number added to the
appendix at camera-ready: intervals and ties for the confirmatory contrasts,
which audit criterion removes the Amylase result, how often each selector
deploys a constant predictor, the λmax floor of collapsed networks, the
training-mean baseline, the epoch 50 to 100 training-loss change, audit
retention, the Holm-family sensitivity analysis, and the NDCG definition. It
reads only committed results and trains nothing.

---

## 7. Conditions

Seven pre-registered conditions plus one exploratory:

| Condition | Benchmark | Shift |
|---|---|---|
| `lipo_random` / `lipo_scaffold` | TDC Lipophilicity (AstraZeneca) | random / Bemis–Murcko scaffold |
| `caco2_random` / `caco2_scaffold` | TDC Caco-2 (Wang) | random / Bemis–Murcko scaffold |
| `cond7` | TDC Lipophilicity (AstraZeneca) | scaffold-disjoint test; train and validation re-split at random, so validation is in-distribution ("Lipophilicity mismatch" in the paper) |
| `amylase` | FLIP2 amylase | close → far homology |
| `hydro` | FLIP2 hydrophobic core (57-residue SH3 domain) | low → high |
| `gdsc_drug` | TDC GDSC2 | leave-drug-out (exploratory) |

---

## 8. License

MIT (see `pyproject.toml`).
