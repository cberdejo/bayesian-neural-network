from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ABCSSConfig:
    """Configuration for ABC Subset Simulation."""

    layer_sizes: list[int] = field(default_factory=lambda: [1, 5, 5, 1])
    activations: list[str] = field(default_factory=lambda: ["tanh", "tanh", "tanh"])
    n_samples: int = 100_000
    sim_levels: int = 6
    p0: float = 0.2
    initial_std: float | None = None
    prior_low: float = -1.0
    prior_high: float = 1.0
    seed: int | None = 42
