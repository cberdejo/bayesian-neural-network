from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VIBBConfig:
    """Configuration for the Variational Inference (Bayes by Backprop) model."""

    layer_sizes: list[int] = field(default_factory=lambda: [1, 15, 15, 1])
    activation: str = "tanh"
    prior_sigma_1: float = 1.5
    prior_sigma_2: float = 0.1
    prior_pi: float = 0.5
    kl_weight: float = 1.0
    noise_std: float = 1.0
    lr: float = 0.08
    epochs: int = 1500
    mc_samples: int = 500
    seed: int | None = 42
    device: str | None = None
