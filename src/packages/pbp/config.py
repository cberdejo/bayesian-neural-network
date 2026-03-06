from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PBPConfig:
    """Configuration for the Probabilistic Backpropagation model."""

    layer_sizes: list[int] = field(default_factory=lambda: [1, 5, 5, 1])
    n_epochs: int = 40
    normalize: bool = True
    seed: int | None = 7
