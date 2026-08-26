"""Dataset loading, split construction, freezing and hashing (plan §2, §9).

Conditions
----------
tdc_random / tdc_scaffold : Lipophilicity, Caco2 via PyTDC (frac 70/10/20)
cond7                     : deliberate mismatch — scaffold-split train+valid merged and
                            re-split RANDOMLY, test scaffolds untouched. Validation
                            becomes ID-like while test stays scaffold-OOD.
flip                      : Amylase close_to_far, Hydro low_to_high; the benchmark's
                            own `set` / `validation` columns are used VERBATIM.
"""
from __future__ import annotations
import os
import hashlib, json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "artifacts/data" / "raw"
CACHE = ROOT / "artifacts/data" / "features"
SPLITS = ROOT / "artifacts/data" / "splits"
FRAC = [0.7, 0.1, 0.2]

TDC_NAMES = {"lipo": "Lipophilicity_AstraZeneca", "caco2": "Caco2_Wang"}
FLIP_FILES = {"amylase": "amylase/close_to_far.csv.gz", "hydro": "hydro/low_to_high.csv.gz"}


@dataclass
class Split:
    Xtr: np.ndarray; ytr: np.ndarray
    Xva: np.ndarray; yva: np.ndarray
    Xte: np.ndarray; yte: np.ndarray
    y_mu: float; y_sigma: float
    meta: dict

    def standardized(self) -> "Split":
        return self  # y already standardized on construction


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------- molecules
def _tdc_frame(key: str) -> pd.DataFrame:
    from tdc.single_pred import ADME
    cwd = Path.cwd()
    RAW.mkdir(parents=True, exist_ok=True)
    import os; os.chdir(RAW)
    try:
        return ADME(name=TDC_NAMES[key]).get_data()
    finally:
        os.chdir(cwd)


def _tdc_split(key: str, method: str, seed: int) -> dict[str, pd.DataFrame]:
    from tdc.single_pred import ADME
    import os
    cwd = Path.cwd(); os.chdir(RAW)
    try:
        return ADME(name=TDC_NAMES[key]).get_split(method=method, seed=seed, frac=FRAC)
    finally:
        os.chdir(cwd)


def _morgan_cached(key: str, smiles: list[str]) -> np.ndarray:
    from .features import morgan
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"morgan_{key}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        if list(z["smiles"]) == list(smiles):
            return z["X"]
    X = morgan(smiles)
    np.savez_compressed(f, X=X, smiles=np.array(smiles, dtype=object))
    return X


def load_molecular(key: str, condition: str, seed: int) -> Split:
    """condition in {'random','scaffold','cond7'}"""
    if condition == "cond7":
        sp = _tdc_split(key, "scaffold", seed)
        pool = pd.concat([sp["train"], sp["valid"]], ignore_index=True)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(pool)); nval = len(sp["valid"])
        va, tr, te = pool.iloc[idx[:nval]], pool.iloc[idx[nval:]], sp["test"]
    else:
        sp = _tdc_split(key, condition, seed)
        tr, va, te = sp["train"], sp["valid"], sp["test"]
    allsmi = list(tr["Drug"]) + list(va["Drug"]) + list(te["Drug"])
    X = _morgan_cached(f"{key}_{condition}_{seed}", allsmi)
    n1, n2 = len(tr), len(tr) + len(va)
    ytr = tr["Y"].to_numpy(np.float64)
    mu, sd = float(ytr.mean()), float(ytr.std() + 1e-12)
    z = lambda a: ((a - mu) / sd).astype(np.float32)
    return Split(X[:n1], z(ytr), X[n1:n2], z(va["Y"].to_numpy(np.float64)),
                 X[n2:], z(te["Y"].to_numpy(np.float64)), mu, sd,
                 meta=dict(dataset=key, condition=condition, seed=seed, d_in=X.shape[1],
                           n_train=n1, n_val=n2 - n1, n_test=len(te)))


# ---------------------------------------------------------------- proteins
def load_flip(key: str, seed: int) -> Split:
    """FLIP splits are FIXED biologically; `seed` varies only HP/init/order."""
    from .features import protein_onehot, FLIP_VOCAB
    df = pd.read_csv(RAW / FLIP_FILES[key])
    tr = df[(df.set == "train") & (~df.validation)]
    va = df[(df.set == "train") & (df.validation)]
    te = df[df.set == "test"]
    assert len(set(tr.index) & set(va.index)) == 0
    assert len(set(tr.index) & set(te.index)) == 0
    assert len(set(va.index) & set(te.index)) == 0
    L = int(df.sequence.str.len().max())
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"onehot_{key}.npz"
    if f.exists():
        X = np.load(f)["X"]
    else:
        X = protein_onehot(list(df.sequence), max_len=L)
        np.savez_compressed(f, X=X)
    Xtr, Xva, Xte = X[tr.index.to_numpy()], X[va.index.to_numpy()], X[te.index.to_numpy()]
    ytr = tr.target.to_numpy(np.float64)
    mu, sd = float(ytr.mean()), float(ytr.std() + 1e-12)
    z = lambda a: ((a - mu) / sd).astype(np.float32)
    return Split(Xtr, z(ytr), Xva, z(va.target.to_numpy(np.float64)),
                 Xte, z(te.target.to_numpy(np.float64)), mu, sd,
                 meta=dict(dataset=key, condition="flip", seed=seed, d_in=X.shape[1],
                           n_train=len(tr), n_val=len(va), n_test=len(te), max_len=L))



# ---------------------------------------------------------- memory-safe expression
# GDSC stores a 17,737-gene vector per row, but every loader uses at most ~1,000 of
# them. Materialising the full matrix costs ~1.3 GB per worker; with several workers
# that pushed a 19 GB laptop into swap. These helpers stream over rows in chunks so
# peak memory is (chunk x 17,737) rather than (n_rows x 17,737), and then keep only
# the selected gene columns.
#
# Gene selection must not change: `_gene_var_chunked` is verified against
# `np.stack(...).var(0)` to select an IDENTICAL gene set, and `_extract_genes` is
# verified bit-identical to `np.stack(...)[:, genes]`.
EXPR_CHUNK = 512


def _gene_var_chunked(series, chunk: int = EXPR_CHUNK) -> np.ndarray:
    """Per-gene variance over rows without materialising the full matrix."""
    arr = series.to_numpy() if hasattr(series, "to_numpy") else np.asarray(series)
    n = len(arr)
    tot = None
    for i in range(0, n, chunk):
        B = np.stack(arr[i:i + chunk])
        tot = B.sum(0) if tot is None else tot + B.sum(0)
    mean = tot / n
    ss = None
    for i in range(0, n, chunk):
        B = np.stack(arr[i:i + chunk])
        d = B - mean
        acc = np.einsum("ij,ij->j", d, d)
        ss = acc if ss is None else ss + acc
    return ss / n


def _extract_genes(series, genes: np.ndarray, chunk: int = EXPR_CHUNK) -> np.ndarray:
    """Stack ONLY the selected gene columns, chunk by chunk."""
    arr = series.to_numpy() if hasattr(series, "to_numpy") else np.asarray(series)
    out = np.empty((len(arr), len(genes)), dtype=np.float64)
    for i in range(0, len(arr), chunk):
        out[i:i + chunk] = np.stack(arr[i:i + chunk])[:, genes]
    return out


# ---------------------------------------------------------------- drug response
# GDSC2 (TDC `DrugRes`): (drug, cancer cell line) -> ln(IC50).
#
# EXPLORATORY, POST-HOC (plan section 9): added after the pre-registered protocol was
# unblinded, in response to venue-fit review. It is NOT part of the confirmatory
# family and must never be pooled with the pre-registered conditions.
#
# Why it earns a place: the ID-val / OOD-test structure here is *natural*, not
# constructed as in `cond7`. A practitioner validates on held-out (drug, cell line)
# pairs drawn from compounds already screened -- in-distribution -- and then deploys
# the model on a brand new compound. Between-drug variation in ln(IC50) spans 3.9
# global standard deviations, comparable to `hydro`'s +4.59 train-sigma.
#
# Cost control: cell lines are subsampled and the transcriptome is reduced to the
# most variable genes. BOTH the gene selection and the z-scoring are fit on TRAINING
# ROWS ONLY, so no validation or test information reaches the feature pipeline.
GDSC_N_CELL_LINES = 80      # of 805
GDSC_N_GENES = 500          # of 17,737
GDSC_N_BITS = 1024          # Morgan bits for the drug
GDSC_TEST_DRUG_FRAC = 0.20
GDSC_VAL_FRAC = 0.125       # of the train-drug pairs -> ~10% overall


def load_gdsc(seed: int, shift: str = "drug") -> Split:
    """`shift='drug'`  : test = held-out COMPOUNDS   (deploy on a new drug)
       `shift='cell'`  : test = held-out CELL LINES  (deploy on a new tumour)

    Validation is always drawn at random from the *training* entities, so it is
    in-distribution by construction while the test set is not.
    """
    from .features import morgan
    from tdc.multi_pred import DrugRes

    df = DrugRes(name="gdsc2").get_data().rename(
        columns={"Cell Line_ID": "cl", "Cell Line": "expr"})
    rng = np.random.default_rng(seed)

    # --- subsample cell lines (cost control), deterministic in `seed` ---
    cls = np.sort(df.cl.unique())
    keep = set(rng.choice(cls, size=min(GDSC_N_CELL_LINES, len(cls)), replace=False))
    df = df[df.cl.isin(keep)].reset_index(drop=True)

    # --- partition the shifted entity into deploy / develop ---
    ent = df.cl if shift == "cell" else df.Drug_ID
    uniq = np.sort(ent.unique())
    n_te = max(1, int(round(GDSC_TEST_DRUG_FRAC * len(uniq))))
    te_ent = set(rng.choice(uniq, size=n_te, replace=False))
    is_te = ent.isin(te_ent).to_numpy()

    dev = df[~is_te].reset_index(drop=True)
    te = df[is_te].reset_index(drop=True)
    # validation is a RANDOM slice of the development pairs -> ID-like
    perm = rng.permutation(len(dev))
    n_va = int(round(GDSC_VAL_FRAC * len(dev)))
    va_idx, tr_idx = perm[:n_va], perm[n_va:]
    tr, va = dev.iloc[tr_idx], dev.iloc[va_idx]

    # --- expression: select + standardise on TRAINING ROWS ONLY ---
    E_tr = np.stack(tr.expr.to_numpy())
    var = E_tr.var(axis=0)
    genes = np.argsort(var)[::-1][:GDSC_N_GENES].copy()
    mu_g, sd_g = E_tr[:, genes].mean(0), E_tr[:, genes].std(0) + 1e-8
    ex = lambda d: ((np.stack(d.expr.to_numpy())[:, genes] - mu_g) / sd_g).astype(np.float32)

    # --- drug: Morgan fingerprint, cached per unique SMILES ---
    smi = {s: i for i, s in enumerate(sorted(df.Drug.unique()))}
    M = morgan(list(smi.keys()), n_bits=GDSC_N_BITS).astype(np.float32)
    fp = lambda d: M[[smi[s] for s in d.Drug]]

    feat = lambda d: np.hstack([fp(d), ex(d)]).astype(np.float32)
    Xtr, Xva, Xte = feat(tr), feat(va), feat(te)

    ytr = tr.Y.to_numpy(np.float64)
    mu, sd = float(ytr.mean()), float(ytr.std() + 1e-12)
    z = lambda a: ((a - mu) / sd).astype(np.float32)
    return Split(Xtr, z(ytr), Xva, z(va.Y.to_numpy(np.float64)),
                 Xte, z(te.Y.to_numpy(np.float64)), mu, sd,
                 meta=dict(dataset="gdsc2", condition=f"gdsc_{shift}", seed=seed,
                           d_in=Xtr.shape[1], n_train=len(tr), n_val=len(va),
                           n_test=len(te), n_cell_lines=len(keep),
                           n_held_out_entities=n_te, exploratory=True))



# ------------------------------------------------- controlled shift severity
# WHY THIS EXISTS (plan section 8.6, exploratory).
#
# Earlier pilot work found that a train-only proxy DOES beat validation -- but only
# on a SYNTHETIC feature-extrapolation task whose shift we had engineered. On real
# covariate shift the same method was neutral. That contrast is the honest
# motivation for this whole study, and it is only convincing if the severity knob
# is explicit and swept.
#
# `alpha` interpolates between an IID split and pure feature extrapolation while
# holding split SIZES fixed, so severity is the only thing that varies:
#
#     score_i = alpha * z(w . x_i) + (1 - alpha) * eps_i,   eps ~ N(0,1)
#     test    = the top 20% by score
#     val     = a random 12.5% of the remainder  -> in-distribution by construction
#
# alpha = 0.0 -> random split (test is IID; val and test are exchangeable)
# alpha = 1.0 -> test is the extreme tail along the leading feature direction
#
# The direction `w` is the first principal component of the pooled features. Using
# pooled data to CONSTRUCT a split is standard (scaffold splitting inspects every
# molecule too); no target information is used, and all target standardisation and
# all model fitting remain strictly train-only.
SHIFT_TEST_FRAC = 0.20
SHIFT_VAL_FRAC = 0.125


def load_shift_severity(key: str, alpha: float, seed: int) -> Split:
    """Severity-controlled feature-extrapolation split of a TDC molecular dataset."""
    df = _tdc_frame(key)
    X = _morgan_cached(f"{key}_pool", list(df["Drug"]))
    y = df["Y"].to_numpy(np.float64)
    rng = np.random.default_rng(seed)

    # leading direction of the pooled feature cloud
    Xc = X - X.mean(0, keepdims=True)
    # randomised power iteration -> first PC without forming the covariance matrix
    w = rng.normal(size=X.shape[1])
    for _ in range(20):
        w = Xc.T @ (Xc @ w)
        w /= np.linalg.norm(w) + 1e-12
    proj = Xc @ w
    proj = (proj - proj.mean()) / (proj.std() + 1e-12)

    score = alpha * proj + (1.0 - alpha) * rng.normal(size=len(proj))
    order = np.argsort(-score)                      # most extreme first
    n_te = int(round(SHIFT_TEST_FRAC * len(order)))
    te_idx = order[:n_te]
    rest = rng.permutation(order[n_te:])
    n_va = int(round(SHIFT_VAL_FRAC * len(rest)))
    va_idx, tr_idx = rest[:n_va], rest[n_va:]

    ytr = y[tr_idx]
    mu, sd = float(ytr.mean()), float(ytr.std() + 1e-12)
    z = lambda a: ((a - mu) / sd).astype(np.float32)
    return Split(X[tr_idx], z(ytr), X[va_idx], z(y[va_idx]),
                 X[te_idx], z(y[te_idx]), mu, sd,
                 meta=dict(dataset=key, condition=f"shiftsev_{key}_a{int(round(alpha*100)):03d}",
                           seed=seed, alpha=alpha, d_in=X.shape[1], n_train=len(tr_idx),
                           n_val=len(va_idx), n_test=len(te_idx), exploratory=True))



# ------------------------------------------------- extreme single-feature shift
# Requested stress test: pick ONE feature, sort the data by it, and put the extreme
# tail in test. This is deliberately harsher than `shiftsev` (§ load_shift_severity),
# which uses an unsupervised principal direction and a severity dial.
#
# WHY NOT "the feature most correlated with test MSE". That would be circular: test
# MSE only exists after a model is trained on a split, so using it to *define* the
# split leaks the outcome into the design, and any resulting validation failure is
# something we manufactured against models we had already fitted. We use the feature
# most correlated with the TARGET instead. This is still adversarial -- it is chosen
# with label knowledge and produces both a covariate and a label shift -- but it is
# constructible before any model exists, so the measurement means something.
#
# The split is defined on RAW expression, before any feature engineering, so the
# train-only gene selection and z-scoring in `load_gdsc` remain uncontaminated.
FEATSHIFT_TEST_FRAC = 0.20
FEATSHIFT_VAL_FRAC = 0.125


def load_feature_shift(seed: int, direction: str = "high") -> Split:
    """GDSC2 split on the single raw gene most correlated with ln(IC50).

    `direction='high'` sends the highest-expressing 20% to test, `'low'` the lowest.
    Train/val are drawn from the remaining 80%, val at random so it stays ID.
    """
    from .features import morgan
    from tdc.multi_pred import DrugRes

    df = DrugRes(name="gdsc2").get_data().rename(
        columns={"Cell Line_ID": "cl", "Cell Line": "expr"})
    rng = np.random.default_rng(seed)
    cls = np.sort(df.cl.unique())
    keep = set(rng.choice(cls, size=min(GDSC_N_CELL_LINES, len(cls)), replace=False))
    df = df[df.cl.isin(keep)].reset_index(drop=True)

    E = np.stack(df.expr.to_numpy())                      # raw, 17,737 genes
    y = df.Y.to_numpy(np.float64)
    # single most target-correlated gene, computed on the pooled data
    Ec = E - E.mean(0, keepdims=True)
    yc = y - y.mean()
    denom = (np.sqrt((Ec ** 2).sum(0)) * np.sqrt((yc ** 2).sum())) + 1e-12
    corr = (Ec.T @ yc) / denom
    gene = int(np.argmax(np.abs(corr)))
    v = E[:, gene]
    order = np.argsort(-v if direction == "high" else v)   # extreme tail first

    n_te = int(round(FEATSHIFT_TEST_FRAC * len(order)))
    te_idx = order[:n_te]
    rest = rng.permutation(order[n_te:])
    n_va = int(round(FEATSHIFT_VAL_FRAC * len(rest)))
    va_idx, tr_idx = rest[:n_va], rest[n_va:]

    # features built AFTER the split, train-only, exactly as load_gdsc does
    E_tr = E[tr_idx]
    var = E_tr.var(axis=0)
    genes = np.argsort(var)[::-1][:GDSC_N_GENES].copy()
    mu_g, sd_g = E_tr[:, genes].mean(0), E_tr[:, genes].std(0) + 1e-8
    # Keep ONLY the selected genes and free the full 17,737-gene array. The closure
    # below would otherwise hold ~1.3 GB per worker for 500 genes' worth of data.
    # E[idx][:, genes] == E[:, genes][idx], so this is arithmetically identical.
    E_sel = E[:, genes]
    del E, E_tr
    ex = lambda idx: ((E_sel[idx] - mu_g) / sd_g).astype(np.float32)
    smi = {t: i for i, t in enumerate(sorted(df.Drug.unique()))}
    M = morgan(list(smi.keys()), n_bits=GDSC_N_BITS).astype(np.float32)
    fp = lambda idx: M[[smi[t] for t in df.Drug.to_numpy()[idx]]]
    feat = lambda idx: np.hstack([fp(idx), ex(idx)]).astype(np.float32)

    ytr = y[tr_idx]
    mu, sd = float(ytr.mean()), float(ytr.std() + 1e-12)
    z = lambda a: ((a - mu) / sd).astype(np.float32)
    return Split(feat(tr_idx), z(ytr), feat(va_idx), z(y[va_idx]),
                 feat(te_idx), z(y[te_idx]), mu, sd,
                 meta=dict(dataset="gdsc2", condition=f"featshift_{direction}", seed=seed,
                           d_in=GDSC_N_BITS + GDSC_N_GENES, n_train=len(tr_idx),
                           n_val=len(va_idx), n_test=len(te_idx), split_gene=gene,
                           split_gene_corr=float(corr[gene]), exploratory=True))



# ------------------------------------------- ADVERSARIAL test-error shift
# ⚠ CIRCULAR BY DESIGN — READ BEFORE USING ANY NUMBER FROM THIS CONDITION.
#
# The split is built FROM model test error. Examples that trained models find hard
# are pushed into the test set; easy ones stay in train/val. Validation is therefore
# guaranteed to be optimistic, and any "validation fails" result here is something we
# manufactured, not something we discovered.
#
# It supports exactly ONE claim, an upper bound:
#     if an adversary builds the worst validation set they can, does a train-only
#     geometry signal recover the right model?
# If the proxy helps here it has a niche at extreme validation failure. If it does
# not help even here, that is the strongest form of the negative result -- no
# realistic shift can be worse than a split engineered against us.
#
# Construction (literal "feature correlated with test MSE"):
#   1. take the 48 x n_test per-example predictions ALREADY stored for `gdsc_drug`
#      (20 outer runs, real trained models -- no probe retraining)
#   2. per-example mean squared error across those models = "hardness"
#   3. ridge-regress hardness on the features -> direction w_err, i.e. the feature
#      combination most correlated with test MSE
#   4. score every pair in the dataset by w_err and send the HARDEST tail to test
#   5. validation is drawn at random from the easy remainder, so it stays ID and
#      maximally optimistic
ADVERR_TEST_FRAC = 0.20
ADVERR_VAL_FRAC = 0.125
ADVERR_RIDGE = 1.0


def load_adverr(seed: int, extreme: bool = True) -> Split:
    """GDSC2 split along the direction most correlated with model test error."""
    from .features import morgan
    from tdc.multi_pred import DrugRes

    base = load_gdsc(seed, "drug")                      # for the feature pipeline
    pf = sorted((ROOT / "artifacts/results" / "exploratory" / "gdsc" / "phase1a" /
                 "predictions" / "gdsc_drug").glob("*.npz"))
    if not pf:
        raise FileNotFoundError("adverr needs the stored gdsc_drug predictions")
    # per-example hardness on the ORIGINAL test split. Each outer run subsamples a
    # different set of cell lines, so test sizes differ; keep only runs whose split
    # matches `base` (same seed -> same subsample -> same n_test).
    per = [((np.load(f)["test_preds"] - base.yte[None, :]) ** 2).mean(0)
           for f in pf if np.load(f)["test_preds"].shape[1] == len(base.yte)]
    if not per:
        raise RuntimeError("no stored gdsc_drug prediction file matches this split")
    hard = np.mean(per, axis=0)

    # direction most correlated with test error (ridge on the original test features)
    Xte = base.Xte.astype(np.float64)
    Xc = Xte - Xte.mean(0, keepdims=True)
    hc = hard - hard.mean()
    G = Xc.T @ Xc + ADVERR_RIDGE * np.eye(Xc.shape[1])
    w_err = np.linalg.solve(G, Xc.T @ hc)
    w_err /= np.linalg.norm(w_err) + 1e-12

    # rebuild the full pool and score every pair by predicted hardness
    df = DrugRes(name="gdsc2").get_data().rename(
        columns={"Cell Line_ID": "cl", "Cell Line": "expr"})
    rng = np.random.default_rng(seed)
    cls = np.sort(df.cl.unique())
    keep = set(rng.choice(cls, size=min(GDSC_N_CELL_LINES, len(cls)), replace=False))
    df = df[df.cl.isin(keep)].reset_index(drop=True)
    E = np.stack(df.expr.to_numpy())
    y = df.Y.to_numpy(np.float64)

    # features must match base's pipeline to apply w_err: use base's gene set
    genes = base.meta["_genes"] if "_genes" in base.meta else None
    if genes is None:                                    # recompute identically
        n_all = len(df)
        idx_all = np.arange(n_all)
        var = E.var(axis=0)
        genes = np.argsort(var)[::-1][:GDSC_N_GENES].copy()
    mu_g, sd_g = E[:, genes].mean(0), E[:, genes].std(0) + 1e-8
    smi = {t: i for i, t in enumerate(sorted(df.Drug.unique()))}
    M = morgan(list(smi.keys()), n_bits=GDSC_N_BITS).astype(np.float32)
    Xall = np.hstack([M[[smi[t] for t in df.Drug]],
                      (E[:, genes] - mu_g) / sd_g]).astype(np.float32)
    score = Xall.astype(np.float64) @ w_err              # predicted hardness

    order = np.argsort(-score)                           # hardest first
    n_te = int(round(ADVERR_TEST_FRAC * len(order)))
    te_idx = order[:n_te]
    rest = order[n_te:]
    if extreme:
        rest_sorted = rest[np.argsort(score[rest])]      # easiest first
        n_va = int(round(ADVERR_VAL_FRAC * len(rest)))
        va_idx = rest_sorted[:n_va]                      # EASIEST -> validation
        tr_idx = rest_sorted[n_va:]
    else:
        rest = rng.permutation(rest)
        n_va = int(round(ADVERR_VAL_FRAC * len(rest)))
        va_idx, tr_idx = rest[:n_va], rest[n_va:]

    E_tr = E[tr_idx]
    g2 = np.argsort(E_tr.var(axis=0))[::-1][:GDSC_N_GENES].copy()
    m2, s2 = E_tr[:, g2].mean(0), E_tr[:, g2].std(0) + 1e-8
    feat = lambda idx: np.hstack([M[[smi[t] for t in df.Drug.to_numpy()[idx]]],
                                  (E[idx][:, g2] - m2) / s2]).astype(np.float32)
    ytr = y[tr_idx]
    mu, sd = float(ytr.mean()), float(ytr.std() + 1e-12)
    z = lambda v: ((v - mu) / sd).astype(np.float32)
    return Split(feat(tr_idx), z(ytr), feat(va_idx), z(y[va_idx]),
                 feat(te_idx), z(y[te_idx]), mu, sd,
                 meta=dict(dataset="gdsc2", condition="adverr", seed=seed,
                           d_in=GDSC_N_BITS + GDSC_N_GENES, n_train=len(tr_idx),
                           n_val=len(va_idx), n_test=len(te_idx),
                           circular_by_design=True, exploratory=True))




# ------------------------------------------------- GDSC with L1000 landmark genes
# Our default GDSC condition reduces 17,737 genes to the top 500 by training-set
# variance. Variance ranking is standard in the drug-response literature but is an
# arbitrary cutoff and an unsupervised heuristic. This variant instead uses the
# **L1000 landmark gene set** -- the ~1,000 genes the LINCS consortium selected as
# sufficient to reconstruct the rest of the transcriptome, and the field's canonical
# dimensionality reduction for expression data.
#
# Provenance: landmark map from the D-GEX release (Chen et al.), 943 symbols; 896 of
# them are present in TDC's GDSC panel. The canonical count is 978 -- this file is a
# subset, so we report 896/943 rather than claiming the full landmark set.
#
# The point of this condition is a SENSITIVITY CHECK: do the selection conclusions
# depend on which genes we kept, or on how many?
L1000_MAP = "l1000/landmark_map.tsv"


def _l1000_symbols() -> set:
    import csv
    out = set()
    with open(RAW / L1000_MAP) as f:
        for row in csv.reader(f, delimiter="\t"):
            if row and row[0]:
                out.add(row[0].strip())
    return out


def load_gdsc_l1000(seed: int) -> Split:
    """GDSC2 held-out-compound split, features = Morgan + L1000 landmark genes."""
    from .features import morgan
    from tdc.multi_pred import DrugRes

    src = DrugRes(name="gdsc2")
    df = src.get_data().rename(columns={"Cell Line_ID": "cl", "Cell Line": "expr"})
    symbols = np.asarray(src.get_gene_symbols())
    lm = _l1000_symbols()
    genes = np.array([i for i, g in enumerate(symbols) if g in lm], dtype=int)

    rng = np.random.default_rng(seed)
    cls = np.sort(df.cl.unique())
    keep = set(rng.choice(cls, size=min(GDSC_N_CELL_LINES, len(cls)), replace=False))
    df = df[df.cl.isin(keep)].reset_index(drop=True)

    # identical split rule to load_gdsc: hold out whole compounds
    uniq = np.sort(df.Drug_ID.unique())
    n_te = max(1, int(round(GDSC_TEST_DRUG_FRAC * len(uniq))))
    te_ent = set(rng.choice(uniq, size=n_te, replace=False))
    is_te = df.Drug_ID.isin(te_ent).to_numpy()
    dev = df[~is_te].reset_index(drop=True); te = df[is_te].reset_index(drop=True)
    perm = rng.permutation(len(dev)); n_va = int(round(GDSC_VAL_FRAC * len(dev)))
    tr, va = dev.iloc[perm[n_va:]], dev.iloc[perm[:n_va]]

    E_tr = np.stack(tr.expr.to_numpy())[:, genes]
    mu_g, sd_g = E_tr.mean(0), E_tr.std(0) + 1e-8      # train-only standardisation
    ex = lambda d: ((np.stack(d.expr.to_numpy())[:, genes] - mu_g) / sd_g).astype(np.float32)
    smi = {t: i for i, t in enumerate(sorted(df.Drug.unique()))}
    M = morgan(list(smi.keys()), n_bits=GDSC_N_BITS).astype(np.float32)
    feat = lambda d: np.hstack([M[[smi[t] for t in d.Drug]], ex(d)]).astype(np.float32)

    ytr = tr.Y.to_numpy(np.float64)
    mu, sd = float(ytr.mean()), float(ytr.std() + 1e-12)
    z = lambda a: ((a - mu) / sd).astype(np.float32)
    Xtr = feat(tr)
    return Split(Xtr, z(ytr), feat(va), z(va.Y.to_numpy(np.float64)),
                 feat(te), z(te.Y.to_numpy(np.float64)), mu, sd,
                 meta=dict(dataset="gdsc2", condition="gdsc_l1000", seed=seed,
                           d_in=Xtr.shape[1], n_train=len(tr), n_val=len(va),
                           n_test=len(te), n_landmark_genes=len(genes),
                           exploratory=True))


CONDITIONS = {
    "lipo_random":   lambda s: load_molecular("lipo", "random", s),
    "lipo_scaffold": lambda s: load_molecular("lipo", "scaffold", s),
    "caco2_random":  lambda s: load_molecular("caco2", "random", s),
    "caco2_scaffold":lambda s: load_molecular("caco2", "scaffold", s),
    "cond7":         lambda s: load_molecular("lipo", "cond7", s),
    "amylase":       lambda s: load_flip("amylase", s),
    "hydro":         lambda s: load_flip("hydro", s),
    # exploratory, post-hoc (see load_gdsc docstring)
    "gdsc_drug":     lambda s: load_gdsc(s, "drug"),
    "gdsc_cell":     lambda s: load_gdsc(s, "cell"),
    # extreme single-feature split on the most target-correlated raw gene
    "adverr":        lambda s: load_adverr(s, extreme=True),   # CIRCULAR by design
    "gdsc_l1000":    lambda s: load_gdsc_l1000(s),   # L1000 landmark genes
    "featshift_high": lambda s: load_feature_shift(s, "high"),
    "featshift_low":  lambda s: load_feature_shift(s, "low"),
    # severity sweep: alpha=0 is IID, alpha=1 is pure feature extrapolation
    **{f"shiftsev_{k}_a{a:03d}": (lambda k=k, a=a: (lambda s: load_shift_severity(k, a/100, s)))()
       for k in ("caco2", "lipo") for a in (0, 25, 50, 75, 100)},
}


def load(condition: str, seed: int) -> Split:
    return CONDITIONS[condition](seed)


def data_hash() -> dict:
    """Hash of every raw input file — pinned in protocol/ before any run (§9)."""
    out = {}
    for p in sorted(RAW.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(RAW))] = _file_sha(p)
    return out


def fold_val_into_train(sp: "Split") -> "Split":
    """Val-free ablation (plan §8.4): train on Xtr u Xva, no held-out validation.
    Targets keep the ORIGINAL train-only standardisation so losses stay comparable."""
    import numpy as _np
    return Split(_np.vstack([sp.Xtr, sp.Xva]), _np.concatenate([sp.ytr, sp.yva]),
                 sp.Xva[:0], sp.yva[:0], sp.Xte, sp.yte, sp.y_mu, sp.y_sigma,
                 meta={**sp.meta, "n_train": len(sp.Xtr) + len(sp.Xva), "n_val": 0,
                       "valfree": True})
