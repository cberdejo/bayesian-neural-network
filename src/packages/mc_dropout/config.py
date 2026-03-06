from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MCDropoutConfig:
    """Configuration for the Monte Carlo Dropout BNN."""

    layer_sizes: list[int] = field(default_factory=lambda: [1, 10, 10, 2])
    dropout_p: float = 0.5
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    mc_samples: int = 50
    seed: int | None = 7
    device: str | None = None
