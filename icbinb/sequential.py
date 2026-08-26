"""Sequential HPO optimizers for Phase 1B (plan §10).

Nine methods = 3 val baselines + 3 signals x {TPE, HEBO} multi-objective.
Sequential objectives only (no rank-sum driving, plan §10.1); rank-sum returns as the
*deployment rule* (§10.3), applied within-budget to whatever pool the search produced.

Search space matches plan §4 exactly, including the zero-inflated weight decay, encoded
as a categorical use_wd gate so both TPE and HEBO can represent the point mass at 0.
"""
from __future__ import annotations
import numpy as np

LR = (1e-4, 3e-3); WD = (1e-6, 1e-2); DO = (0.0, 0.5)
WIDTHS = [64, 128, 256, 512]; DEPTHS = [1, 2, 3, 4]
SIGNAL = {"fg": "fg_legacy", "penult": "fg_penult", "hess": "hess_top"}


def _hp(lr, use_wd, wd, dropout, width, depth) -> dict:
    return dict(lr=float(lr), weight_decay=float(wd) if int(use_wd) else 0.0,
                dropout=float(dropout), width=int(width), depth=int(depth))


class RandomSearch:
    """Uniform random proposals; selection happens afterwards via the deploy rule."""
    def __init__(self, seed): self.rng = np.random.default_rng(seed)
    def suggest(self):
        r = self.rng
        return _hp(10 ** r.uniform(np.log10(LR[0]), np.log10(LR[1])),
                   r.random() > 0.2, 10 ** r.uniform(np.log10(WD[0]), np.log10(WD[1])),
                   r.uniform(*DO), r.choice(WIDTHS), r.choice(DEPTHS))
    def observe(self, hp, y): pass


class OptunaOpt:
    """TPESampler; single-objective on val, or multi-objective (val, signal)."""
    def __init__(self, seed, n_obj=1):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        # NOTE: do NOT hold a module reference here -- it makes the optimiser
        # unpicklable, which blocks mid-shard checkpointing (see run_phase1b.py).
        sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
        self.study = (optuna.create_study(direction="minimize", sampler=sampler) if n_obj == 1
                      else optuna.create_study(directions=["minimize"]*n_obj, sampler=sampler))
        self._t = None
    def suggest(self):
        t = self.study.ask()
        self._t = t
        return _hp(t.suggest_float("lr", *LR, log=True),
                   t.suggest_categorical("use_wd", [0, 1]),
                   t.suggest_float("wd", *WD, log=True),
                   t.suggest_float("dropout", *DO),
                   t.suggest_categorical("width", WIDTHS),
                   t.suggest_categorical("depth", DEPTHS))
    def observe(self, hp, y):
        self.study.tell(self._t, y if isinstance(y, (list, tuple)) else float(y))


class HeboOpt:
    """HEBO for single objective; GeneralBO(num_obj=2) for multi-objective.

    Verified in Phase 0: the plain `HEBO` class has no `num_obj`, so MO requires
    `hebo.optimizers.general.GeneralBO` (vector LCB + NSGA-II, kappa=2.0 default).
    """
    SPACE = [{"name":"lr","type":"pow","lb":LR[0],"ub":LR[1]},
             {"name":"use_wd","type":"cat","categories":[0,1]},
             {"name":"wd","type":"pow","lb":WD[0],"ub":WD[1]},
             {"name":"dropout","type":"num","lb":DO[0],"ub":DO[1]},
             {"name":"width","type":"cat","categories":WIDTHS},
             {"name":"depth","type":"cat","categories":DEPTHS}]
    def __init__(self, seed, n_obj=1):
        from hebo.design_space.design_space import DesignSpace
        np.random.seed(seed)
        sp = DesignSpace().parse(self.SPACE)
        if n_obj == 1:
            from hebo.optimizers.hebo import HEBO
            self.opt = HEBO(sp, rand_sample=8, scramble_seed=seed)
        else:
            from hebo.optimizers.general import GeneralBO
            self.opt = GeneralBO(sp, num_obj=n_obj, num_constr=0, rand_sample=8)
        self.n_obj = n_obj; self._rec = None
    def suggest(self):
        rec = self.opt.suggest(n_suggestions=1); self._rec = rec
        r = rec.iloc[0]
        return _hp(r["lr"], r["use_wd"], r["wd"], r["dropout"], r["width"], r["depth"])
    def observe(self, hp, y):
        arr = np.array([y], dtype=float).reshape(1, -1) if np.isscalar(y) else np.array(y, float).reshape(1, -1)
        self.opt.observe(self._rec, arr)


def make(method: str, seed: int):
    """method -> (optimizer, objective_key or None for MO signal)."""
    if method == "random_val": return RandomSearch(seed), None
    if method == "tpe_val":    return OptunaOpt(seed, 1), None
    if method == "hebo_val":   return HeboOpt(seed, 1), None
    fam, sig = method.split("_")[0], method.split("_")[1]
    if sig not in SIGNAL: raise ValueError(method)
    opt = OptunaOpt(seed, 2) if fam == "tpe" else HeboOpt(seed, 2)
    return opt, SIGNAL[sig]


METHODS = ["random_val","tpe_val","hebo_val",
           "tpe_fg_mo","hebo_fg_mo","tpe_penult_mo","hebo_penult_mo",
           "tpe_hess_mo","hebo_hess_mo"]
