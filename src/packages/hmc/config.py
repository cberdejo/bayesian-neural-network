from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HMCConfig:
    """Configuration for the Hamiltonian Monte Carlo BNN."""

    layer_sizes: list[int] = field(default_factory=lambda: [1, 10, 10, 1])
    step_size: float = 0.0015
    num_samples: int = 900
    num_steps_per_sample: int = 12
    tau_out: float = 100.0
    tau_prior: float = 1.0
    burn_frac: float = 0.5
    seed: int | None = 42
    device: str | None = None
