"""Selection signals (plan §6.1) — light forward-only path + degeneracy diagnostics.

Tracked signals
---------------
fg_legacy   max_b mean_l ||A_l||_F^2 / (B w)         activation energy
fg_penult   max_b        ||A_L||_F^2 / (B w)         last hidden layer only
hess_top    lambda_max(Hessian of train MSE)         true curvature (fgbo.proxies)

The light path computes ONLY post-activation quantities; it skips the spectral / gate /
LLH / LGN terms of `MLP.compute_representation_proxies` (3 `matrix_norm` calls per layer
per batch), which the tracked signals do not use. Bit-identity with the full path is
asserted in the test suite.

Degeneracy diagnostics (plan §7.1) — the main confounds for an activation-energy proxy:
  dead_relu_frac   fraction of hidden units that are zero for EVERY proxy example.
                   Energy falls with this, so a capacity-lost net can look "flat".
  pred_std         output spread; near-zero flags a collapsed constant predictor,
                   which would otherwise score well on fg_penult.
"""
from __future__ import annotations
import numpy as np
import torch

__all__ = ["forward_signals"]


@torch.no_grad()
def forward_signals(model, batches) -> dict:
    """One forward pass over the fixed proxy batches. `model` must already be eval()."""
    model.eval()
    per_layer_batch: list[list[float]] = []   # [batch][layer]
    alive: list[torch.Tensor] = []            # per layer: any-nonzero mask over samples
    preds: list[torch.Tensor] = []
    for xb in batches:
        x = xb
        layer_e: list[float] = []
        for li, (linear, bn, drop) in enumerate(
                zip(model.hidden_layers, model.bn_layers, model.dropouts)):
            z = bn(linear(x))
            a = drop(model.act(z))
            B, w = a.shape
            layer_e.append(float(a.square().sum().item()) / max(B * w, 1))
            nz = (a != 0).any(dim=0)
            if li >= len(alive): alive.append(nz)
            else: alive[li] = alive[li] | nz
            x = a
        per_layer_batch.append(layer_e)
        preds.append(model.output_layer(x).detach().reshape(-1))

    E = np.asarray(per_layer_batch)                       # (n_batches, depth)
    fg_legacy = float(E.mean(axis=1).max())               # max_b mean_l
    fg_penult = float(E[:, -1].max())                     # max_b, last layer
    energy_per_layer = E.max(axis=0).tolist()             # max over batches, per layer
    dead = [1.0 - float(m.float().mean().item()) for m in alive]
    p = torch.cat(preds)
    return dict(
        fg_legacy=fg_legacy,
        fg_penult=fg_penult,
        energy_per_layer=energy_per_layer,
        dead_relu_frac=float(np.mean(dead)),
        dead_relu_frac_penult=float(dead[-1]),
        pred_std=float(p.std().item()),
        pred_mean=float(p.mean().item()),
        n_proxy_batches=len(batches),
    )
