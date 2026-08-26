"""48-candidate hyperparameter design (plan §5): scrambled Latin hypercube.

Sobol's balance guarantees hold at powers of two and 48 is not one, so we use LHS and
do not claim Sobol coverage properties. Design mechanics:
  * LHS strata on the continuous dims (log-lr, dropout, log-wd)
  * weight decay is ZERO-INFLATED: exactly 20% of candidates get wd == 0.0, applied as
    a stratified (not i.i.d.) mask so the marginal is exact
  * width/depth assigned by stratified permutation so their marginals are exact
"""
from __future__ import annotations
import math
import numpy as np

LR_LO, LR_HI = 1e-4, 3e-3
WD_LO, WD_HI = 1e-6, 1e-2
WD_ZERO_FRAC = 0.20
DROPOUT_LO, DROPOUT_HI = 0.0, 0.5
WIDTHS = (64, 128, 256, 512)
DEPTHS = (1, 2, 3, 4)

def _lhs(n: int, rng: np.random.Generator) -> np.ndarray:
    """One stratified, scrambled LHS column on [0,1)."""
    return (rng.permutation(n) + rng.random(n)) / n

def _stratified_categorical(n: int, levels: tuple, rng: np.random.Generator) -> np.ndarray:
    """Exact-marginal assignment: each level appears n/len(levels) times (+/-1)."""
    reps = int(math.ceil(n / len(levels)))
    pool = np.tile(np.arange(len(levels)), reps)[:n]
    return np.asarray(levels, dtype=object)[rng.permutation(pool)]

def make_candidates(n: int, hp_seed: int) -> list[dict]:
    rng = np.random.default_rng(hp_seed)
    u_lr, u_wd, u_do = _lhs(n, rng), _lhs(n, rng), _lhs(n, rng)
    lr = 10 ** (np.log10(LR_LO) + u_lr * (np.log10(LR_HI) - np.log10(LR_LO)))
    wd = 10 ** (np.log10(WD_LO) + u_wd * (np.log10(WD_HI) - np.log10(WD_LO)))
    dropout = DROPOUT_LO + u_do * (DROPOUT_HI - DROPOUT_LO)
    # exact 20% zeros, stratified (not i.i.d. Bernoulli)
    n_zero = int(round(WD_ZERO_FRAC * n))
    wd[rng.permutation(n)[:n_zero]] = 0.0
    width = _stratified_categorical(n, WIDTHS, rng)
    depth = _stratified_categorical(n, DEPTHS, rng)
    return [dict(lr=float(lr[i]), weight_decay=float(wd[i]), dropout=float(dropout[i]),
                 width=int(width[i]), depth=int(depth[i])) for i in range(n)]
