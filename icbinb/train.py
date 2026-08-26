"""Train one candidate and emit one fully-populated result row (plan §4, §7.1)."""
from __future__ import annotations
import math, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fgbo.models.mlp import MLP, MLPConfig
from fgbo.proxies import top_hessian_eig, hessian_trace
from .proxies import forward_signals

EPOCHS, BATCH, PROXY_CAP = 100, 64, 1024
DIAG_EPOCHS = (25, 50)


TRACE_PROBES = 30      # Hutchinson probes; verified unbiased vs the exact trace


def proxy_subset(n_train: int, proxy_seed: int) -> np.ndarray:
    """Fixed subset, an EXACT multiple of BATCH (no ragged final batch) — plan §6."""
    n = (min(PROXY_CAP, n_train) // BATCH) * BATCH
    return np.random.default_rng(proxy_seed).choice(n_train, n, replace=False)


def _mse(model, X, y) -> float:
    model.eval()
    with torch.no_grad():
        return float(((model(X).squeeze(-1) - y) ** 2).mean().item())


def _metrics(model, X, y):
    from scipy.stats import spearmanr
    model.eval()
    with torch.no_grad():
        p = model(X).squeeze(-1).cpu().numpy()
    t = y.cpu().numpy()
    mse = float(((p - t) ** 2).mean())
    mae = float(np.abs(p - t).mean())
    rho = float(spearmanr(p, t).statistic) if len(t) > 2 and np.std(p) > 0 else float("nan")
    return mse, mae, rho, p


def train_candidate(split, hp: dict, seeds, candidate_id: int, *, epochs: int = EPOCHS,
                    device: str = "cpu", worker_id: int = 0, extra: dict | None = None) -> dict:
    t_start = time.time()
    torch.set_num_threads(1)
    dev = torch.device(device)
    Xtr = torch.from_numpy(split.Xtr).to(dev); ytr = torch.from_numpy(split.ytr).to(dev)
    Xva = torch.from_numpy(split.Xva).to(dev); yva = torch.from_numpy(split.yva).to(dev)
    Xte = torch.from_numpy(split.Xte).to(dev); yte = torch.from_numpy(split.yte).to(dev)

    torch.manual_seed(seeds.init_seed + candidate_id)
    model = MLP(MLPConfig(in_features=Xtr.shape[1], out_features=1,
                          hidden_width=hp["width"], num_layers=hp["depth"],
                          dropout=hp["dropout"], use_batchnorm=False,
                          activation="relu")).to(dev)
    # COUPLED L2 weight decay (plan §4) — Adam, not AdamW
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    crit = nn.MSELoss()
    g = torch.Generator().manual_seed(seeds.loader_seed)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH, shuffle=True, generator=g)

    pidx = proxy_subset(len(Xtr), seeds.proxy_seed)
    Xp, yp = Xtr[pidx], ytr[pidx]
    pbatches = [Xp[i:i + BATCH] for i in range(0, len(Xp), BATCH)]

    diag: dict = {}
    status_train, detail, n_done = "ok", "", 0
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb).squeeze(-1), yb)
            if not torch.isfinite(loss):
                status_train, detail = "diverged", f"nonfinite loss @epoch{ep}"
                break
            loss.backward(); opt.step()
        if status_train != "ok": break
        n_done = ep
        if ep in DIAG_EPOCHS:
            fs = forward_signals(model, pbatches)
            diag[f"train_mse_e{ep}"] = _mse(model, Xtr, ytr)
            diag[f"val_mse_e{ep}"]   = _mse(model, Xva, yva)
            diag[f"fg_legacy_e{ep}"] = fs["fg_legacy"]
            diag[f"fg_penult_e{ep}"] = fs["fg_penult"]
            try:
                diag[f"hess_top_e{ep}"] = top_hessian_eig(model, Xp, yp, seed=seeds.proxy_seed)[0]
            except Exception:
                diag[f"hess_top_e{ep}"] = float("nan")

    row: dict = dict(candidate_id=candidate_id, **hp, **seeds.asdict(),
                     batch_size=BATCH, epochs=epochs, n_epochs_actual=n_done,
                     status_train=status_train, status_detail=detail,
                     device=device, worker_id=worker_id,
                     timestamp_start=t_start, torch_version=torch.__version__,
                     n_params=sum(p.numel() for p in model.parameters()),
                     d_in=int(Xtr.shape[1]), n_train=len(Xtr), n_val=len(Xva), n_test=len(Xte),
                     y_mu=split.y_mu, y_sigma=split.y_sigma,
                     n_proxy=len(pidx), **diag)
    if extra: row.update(extra)
    row["runtime_train_sec"] = time.time() - t_start

    if status_train != "ok":
        row.update(status_proxy="skipped", **{k: float("nan") for k in
                   ("train_mse_std","val_mse_std","test_mse_std","test_mse_raw","test_mae_raw",
                    "val_spearman","test_spearman","fg_legacy","fg_penult","hess_top",
                    "dead_relu_frac","dead_relu_frac_penult","pred_std_train","pred_std_test")})
        return row

    # --- losses / metrics -------------------------------------------------
    tr_mse, _, _, ptr = _metrics(model, Xtr, ytr)
    va_mse, _, va_rho, _ = _metrics(model, Xva, yva)
    te_mse, te_mae, te_rho, pte = _metrics(model, Xte, yte)
    s2 = split.y_sigma ** 2
    row.update(train_mse_std=tr_mse, val_mse_std=va_mse, test_mse_std=te_mse,
               train_mse_raw=tr_mse * s2, val_mse_raw=va_mse * s2, test_mse_raw=te_mse * s2,
               test_mae_raw=te_mae * split.y_sigma,
               val_spearman=va_rho, test_spearman=te_rho,
               pred_std_train=float(ptr.std()), pred_std_test=float(pte.std()))

    # --- signals ----------------------------------------------------------
    t0 = time.time(); status_proxy = "ok"
    try:
        fs = forward_signals(model, pbatches)
        row.update(fg_legacy=fs["fg_legacy"], fg_penult=fs["fg_penult"],
                   energy_per_layer=fs["energy_per_layer"],
                   dead_relu_frac=fs["dead_relu_frac"],
                   dead_relu_frac_penult=fs["dead_relu_frac_penult"],
                   n_proxy_batches=fs["n_proxy_batches"])
    except Exception as e:
        status_proxy = "fg_failed"; detail += f" fg:{e}"
        row.update(fg_legacy=float("nan"), fg_penult=float("nan"))
    row["runtime_proxy_sec"] = time.time() - t0

    t0 = time.time()
    try:
        eig, iters, conv = top_hessian_eig(model, Xp, yp, seed=seeds.proxy_seed)
        row.update(hess_top=eig, hess_iters=iters, hess_converged=conv)
    except Exception as e:
        status_proxy = "both_failed" if status_proxy == "fg_failed" else "hess_failed"
        detail += f" hess:{e}"
        row.update(hess_top=float("nan"), hess_iters=0, hess_converged=False)
    row["runtime_hess_sec"] = time.time() - t0

    # ---- Hessian TRACE (bulk curvature), Hutchinson ----
    # lambda_max summarises ONE direction and is a weak generalisation predictor
    # (Kaur et al. 2022). The trace summarises the whole spectrum, correlates better
    # with the generalisation gap (Petzka et al. 2021) and is what label-noise SGD
    # implicitly minimises. Under Gauss-Newton, tr(H) ~ ||J||_F^2 -- a Frobenius
    # quantity, like the activation-energy proxy -- so the trace, not lambda_max, is
    # the right curvature reference for that proxy. Petzka's relative flatness
    # (||w||^2 * tr H) is formed post hoc from `hess_trace` and `weight_norm_fro`.
    t0 = time.time()
    try:
        tr, tr_se = hessian_trace(model, Xp, yp, n_probes=TRACE_PROBES,
                                  seed=seeds.proxy_seed)
        row.update(hess_trace=tr, hess_trace_se=tr_se)
    except Exception as e:
        status_proxy = "both_failed" if status_proxy != "ok" else "trace_failed"
        detail += f" trace:{e}"
        row.update(hess_trace=float("nan"), hess_trace_se=float("nan"))
    row["runtime_trace_sec"] = time.time() - t0

    W = [m.weight for m in list(model.hidden_layers) + [model.output_layer]]
    with torch.no_grad():
        row["weight_norm_fro"] = float(sum(w.square().sum() for w in W).sqrt().item())
        specs = [float(torch.linalg.matrix_norm(w, ord=2).item()) for w in W]
    row["weight_norm_per_layer"] = specs
    row["weight_norm_spec_prod"] = float(np.prod(specs))
    row["status_proxy"] = status_proxy
    row["status_detail"] = detail
    row["test_preds"] = pte
    return row
