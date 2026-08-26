"""Configurable MLP regressor with activation capture for flatness proxy.

Architecture (per hidden block):
    Linear → [BatchNorm] → Activation → [Dropout]
Final layer: Linear (no activation, no BN, no dropout).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class MLPConfig:
    in_features: int
    out_features: int = 1
    hidden_width: int = 128
    num_layers: int = 3
    dropout: float = 0.0
    use_batchnorm: bool = True
    activation: str = "relu"   # relu | tanh | elu


@dataclass
class RepresentationProxies:
    """Forward-only representation metrics aggregated over a data loader."""

    energy: float = 0.0
    energy_penult: float = 0.0
    gate_instability: float = 0.0
    spectral_concentration: float = 0.0
    scale_invariant: float = 0.0
    train_fast: float = 0.0
    llh_max: float = 0.0
    llh_mean: float = 0.0
    lgn: float = 0.0


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """Fully-connected regressor with optional BatchNorm and Dropout.

    Parameters
    ----------
    cfg : MLPConfig
        Architecture specification.

    Attributes
    ----------
    hidden_layers : nn.ModuleList
        Linear layers for all hidden blocks.
    output_layer : nn.Linear
        Final prediction head.
    """

    def __init__(self, cfg: MLPConfig) -> None:
        super().__init__()
        self.cfg = cfg

        act_cls = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "elu": nn.ELU,
        }[cfg.activation]

        # Build hidden blocks
        self.hidden_layers = nn.ModuleList()
        self.bn_layers: nn.ModuleList = nn.ModuleList()
        self.dropouts: nn.ModuleList = nn.ModuleList()

        in_dim = cfg.in_features
        for _ in range(cfg.num_layers):
            self.hidden_layers.append(nn.Linear(in_dim, cfg.hidden_width))
            self.bn_layers.append(
                nn.BatchNorm1d(cfg.hidden_width) if cfg.use_batchnorm else nn.Identity()
            )
            self.dropouts.append(
                nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
            )
            in_dim = cfg.hidden_width

        self.act = act_cls()
        self.output_layer = nn.Linear(in_dim, cfg.out_features)

        self._init_weights()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for linear, bn, drop in zip(self.hidden_layers, self.bn_layers, self.dropouts):
            x = linear(x)
            x = bn(x)
            x = self.act(x)
            x = drop(x)
        return self.output_layer(x)

    # ------------------------------------------------------------------
    # Parameter count
    # ------------------------------------------------------------------

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Activation capture (for flatness proxy)
    # ------------------------------------------------------------------

    def compute_flatness_proxy(
        self,
        loader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> float:
        """Compute the flatness proxy on *loader*.

        Flatness proxy
        --------------
        Let ``A_l ∈ R^{B × w}`` be the post-activation tensor for layer ``l``
        in a batch of size ``B``, where ``w = hidden_width``.

            score(batch) = mean_l [ ‖A_l‖_F² / (B · w) ]

        The proxy is the **maximum** of ``score`` over all batches in *loader*.
        This measures the typical per-neuron activation energy, which correlates
        with sharpness in parameter space.

        Returns
        -------
        float
            The flatness proxy value (lower ≈ flatter).
        """
        return self.compute_representation_proxies(loader, device).energy

    def compute_representation_proxies(
        self,
        loader: torch.utils.data.DataLoader,
        device: torch.device,
        spectral_alpha: float = 1.0,
        gate_beta: float = 1.0,
        gate_tau: float = 0.1,
        eps: float = 1e-8,
    ) -> RepresentationProxies:
        """Compute cheap forward-only representation proxies.

        ``train_fast`` combines scale-normalised activation energy, spectral
        concentration, and near-boundary activation frequency. All metrics are
        maxima over batches of the mean hidden-layer score.
        """
        self.eval()
        batch_energy: list[float] = []
        batch_energy_penult: list[float] = []
        batch_gate: list[float] = []
        batch_spectral: list[float] = []
        batch_scale_invariant: list[float] = []
        batch_train_fast: list[float] = []
        batch_llh_max: list[float] = []
        batch_llh_mean: list[float] = []
        batch_lgn: list[float] = []

        with torch.no_grad():
            for X, _ in loader:
                x = X.to(device)
                layer_inputs = [x]
                energy_scores: list[float] = []
                gate_scores: list[float] = []
                spectral_scores: list[float] = []
                scale_scores: list[float] = []
                combined_scores: list[float] = []

                for idx, (linear, bn, drop) in enumerate(
                    zip(self.hidden_layers, self.bn_layers, self.dropouts)
                ):
                    z = bn(linear(x))
                    a = drop(self.act(z))
                    batch_size, width = a.shape
                    frob_sq = float(a.square().sum().item())
                    energy = frob_sq / max(batch_size * width, 1)

                    if self.cfg.activation == "relu":
                        threshold = gate_tau
                        if not self.cfg.use_batchnorm:
                            threshold *= float(z.std(unbiased=False).item())
                        gate = float(
                            (z.abs() < max(threshold, eps)).float().mean().item()
                        )
                    else:
                        gate = 0.0

                    if frob_sq > eps:
                        sigma_sq = float(
                            torch.linalg.matrix_norm(a, ord=2).square().item()
                        )
                        spectral = sigma_sq / (frob_sq + eps)
                    else:
                        spectral = 0.0

                    next_weight = (
                        self.hidden_layers[idx + 1].weight
                        if idx + 1 < len(self.hidden_layers)
                        else self.output_layer.weight
                    )
                    scale = torch.sqrt(
                        linear.weight.square().sum()
                        * next_weight.square().sum()
                    ).item()
                    scale_invariant = (frob_sq / max(batch_size, 1)) / (
                        width * scale + eps
                    )
                    combined = (
                        scale_invariant
                        * (1.0 + spectral_alpha * spectral)
                        * (1.0 + gate_beta * gate)
                    )

                    energy_scores.append(energy)
                    gate_scores.append(gate)
                    spectral_scores.append(spectral)
                    scale_scores.append(scale_invariant)
                    combined_scores.append(combined)
                    x = a
                    layer_inputs.append(x)

                if energy_scores:
                    ones = torch.ones(
                        x.shape[0], 1, device=x.device, dtype=x.dtype
                    )
                    augmented = torch.cat([x, ones], dim=1)
                    augmented_frob_sq = float(augmented.square().sum().item())
                    augmented_sigma_sq = float(
                        torch.linalg.matrix_norm(augmented, ord=2).square().item()
                    )
                    batch_llh_max.append(
                        augmented_sigma_sq / max(augmented.shape[0], 1)
                    )
                    batch_llh_mean.append(
                        augmented_frob_sq
                        / max(augmented.shape[0] * augmented.shape[1], 1)
                    )

                    linear_weights = [
                        layer.weight for layer in self.hidden_layers
                    ] + [self.output_layer.weight]
                    spectral_sq = [
                        float(
                            torch.linalg.matrix_norm(weight, ord=2)
                            .square()
                            .item()
                        )
                        for weight in linear_weights
                    ]
                    downstream_products = [1.0] * len(linear_weights)
                    running_product = 1.0
                    for idx in range(len(linear_weights) - 1, -1, -1):
                        downstream_products[idx] = running_product
                        running_product *= spectral_sq[idx]

                    lgn_scores: list[float] = []
                    for activation, downstream in zip(
                        layer_inputs, downstream_products
                    ):
                        activation_augmented = torch.cat(
                            [
                                activation,
                                torch.ones(
                                    activation.shape[0],
                                    1,
                                    device=activation.device,
                                    dtype=activation.dtype,
                                ),
                            ],
                            dim=1,
                        )
                        mean_energy = float(
                            activation_augmented.square().mean().item()
                        )
                        lgn_scores.append(downstream * mean_energy)
                    batch_lgn.append(sum(lgn_scores) / len(lgn_scores))

                    batch_energy.append(sum(energy_scores) / len(energy_scores))
                    # penultimate-layer variant: the SAME per-neuron activation
                    # energy, but read off only the last hidden layer (the input to
                    # the output head) instead of averaged over all layers. For
                    # ``num_layers == 1`` it coincides with ``energy`` by definition.
                    batch_energy_penult.append(energy_scores[-1])
                    batch_gate.append(sum(gate_scores) / len(gate_scores))
                    batch_spectral.append(sum(spectral_scores) / len(spectral_scores))
                    batch_scale_invariant.append(sum(scale_scores) / len(scale_scores))
                    batch_train_fast.append(sum(combined_scores) / len(combined_scores))

        def _safe_max(values: list[float], penalty: float = 1e12) -> float:
            if any(not math.isfinite(value) for value in values):
                return penalty
            return max(values, default=0.0)

        return RepresentationProxies(
            energy=_safe_max(batch_energy),
            energy_penult=_safe_max(batch_energy_penult),
            gate_instability=_safe_max(batch_gate),
            spectral_concentration=_safe_max(batch_spectral),
            scale_invariant=_safe_max(batch_scale_invariant),
            train_fast=_safe_max(batch_train_fast),
            llh_max=_safe_max(batch_llh_max),
            llh_mean=_safe_max(batch_llh_mean),
            lgn=_safe_max(batch_lgn),
        )
