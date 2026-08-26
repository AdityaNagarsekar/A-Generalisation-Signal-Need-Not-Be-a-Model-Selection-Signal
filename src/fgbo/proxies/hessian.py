"""Top Hessian eigenvalue of the training loss — the conventional sharpness measure.

Serves as the *true-curvature* control for the activation-energy proxies: it is what
"sharpness" actually means in the flat-minima literature, so it tests whether the cheap
activation-energy signals are standing in for curvature at all.

Implementation: power iteration with exact Hessian-vector products (Pearlmutter's
trick, i.e. double backward). No external dependency; converges in 5-8 iterations at
``tol=1e-3`` and costs 0.05-0.35 s for the architectures in this study.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

__all__ = ["top_hessian_eig"]


def top_hessian_eig(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    n_iter: int = 30,
    tol: float = 1e-3,
    seed: int = 0,
    loss_fn: nn.Module | None = None,
) -> tuple[float, int, bool]:
    """Largest-magnitude Hessian eigenvalue of the loss on ``(X, y)``.

    Unlike the activation-energy proxies (which are maxima over fixed batches), the
    Hessian is a **single full-subset quantity**: the Hessian of the mean loss is
    linear in the sum over examples, so batching would be an implementation detail
    rather than part of the definition. We therefore evaluate it on the whole proxy
    subset in one pass.

    Power iteration converges to the eigenvalue of largest *absolute* value. For a
    converged network ``lambda_min ~ 0`` so this is ``lambda_max`` (verified against
    exact eigendecomposition); a negative return means ``|lambda_min| > lambda_max``,
    i.e. the model sits at a saddle. That case is flagged rather than silently
    returned, because it is diagnostic of a non-converged candidate.

    Parameters
    ----------
    model
        Network. Put in ``eval()`` mode by the caller's protocol (dropout must be off).
    X, y
        The fixed proxy subset (training examples only).
    n_iter, tol
        Power-iteration cap and relative convergence tolerance.
    seed
        Seeds the starting vector, so the estimate is deterministic across candidates.
    loss_fn
        Defaults to ``nn.MSELoss()`` (the training objective of this study).

    Returns
    -------
    (eigenvalue, iterations_used, converged)
    """
    model.eval()
    crit = nn.MSELoss() if loss_fn is None else loss_fn
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return 0.0, 0, True

    gen = torch.Generator(device="cpu").manual_seed(seed)
    v = [
        torch.randn(p.shape, generator=gen, dtype=p.dtype).to(p.device)
        for p in params
    ]
    norm = torch.sqrt(sum((vi * vi).sum() for vi in v))
    v = [vi / norm for vi in v]

    eig = 0.0
    eig_old: float | None = None
    for it in range(1, n_iter + 1):
        model.zero_grad(set_to_none=True)
        loss = crit(model(X).squeeze(-1), y)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        gv = sum((g * vi).sum() for g, vi in zip(grads, v))
        hv = [h.detach() for h in torch.autograd.grad(gv, params)]

        eig = float(sum((h * vi).sum() for h, vi in zip(hv, v)))
        hv_norm = torch.sqrt(sum((h * h).sum() for h in hv))
        if float(hv_norm) < 1e-12:          # flat direction: Hessian ~ 0
            return eig, it, True
        v = [h / hv_norm for h in hv]

        if eig_old is not None and abs(eig - eig_old) <= tol * max(abs(eig_old), 1e-9):
            return eig, it, True
        eig_old = eig

    return eig, n_iter, False


def hessian_trace(model, X, y, n_probes: int = 20, seed: int = 0,
                  loss_fn=None) -> tuple[float, float]:
    """Hutchinson estimate of tr(H) for the training loss, with its standard error.

    tr(H) = E_v[v^T H v] for v with i.i.d. zero-mean unit-variance entries; we use
    Rademacher probes, which minimise the variance of that estimator. Each probe
    costs one Hessian-vector product, so 20 probes is comparable to the power
    iteration used by `top_hessian_eig`.

    WHY THE TRACE AND NOT ONLY lambda_max. The top eigenvalue summarises a single
    direction and is a poor predictor of generalisation (Kaur et al., 2022). The
    trace summarises the bulk of the spectrum, correlates better with the
    generalisation gap (Petzka et al., 2021, via relative flatness = trace / weight
    norm), and is the quantity implicitly minimised by label-noise SGD (Blanc et al.,
    2020; Li et al., 2021; Damian et al., 2021). Under the Gauss-Newton
    approximation for squared loss, H ~ J^T J and tr(H) ~ ||J||_F^2 -- a Frobenius
    quantity, like the activation-energy proxy, which is why the trace is the
    appropriate curvature reference for it.

    Returns
    -------
    (trace, stderr) : both float; stderr is over the `n_probes` probe estimates.
    """
    import torch

    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    crit = loss_fn if loss_fn is not None else torch.nn.MSELoss()
    loss = crit(model(X).squeeze(-1), y)
    grads = torch.autograd.grad(loss, params, create_graph=True)

    g = torch.Generator().manual_seed(seed)
    ests = []
    for _ in range(n_probes):
        vs = [(torch.randint(0, 2, p.shape, generator=g, dtype=p.dtype) * 2 - 1)
              for p in params]
        hv = torch.autograd.grad(grads, params, grad_outputs=vs, retain_graph=True)
        ests.append(float(sum((h * v).sum() for h, v in zip(hv, vs))))
    a = np.asarray(ests, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a)))
