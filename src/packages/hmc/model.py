from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

import hamiltorch

from .config import HMCConfig


class HMCNet(nn.Module):
    """Simple MLP for use with Hamiltonian Monte Carlo sampling."""

    def __init__(self, layer_sizes: Sequence[int], bias: bool = True):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias=bias))
            if i < len(layer_sizes) - 2:
                layers.append(nn.LeakyReLU(0.1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @classmethod
    def from_config(cls, cfg: HMCConfig) -> "HMCNet":
        return cls(layer_sizes=cfg.layer_sizes)


def sample_hmc(
    net: HMCNet,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: HMCConfig,
) -> list[torch.Tensor]:
    """Run HMC sampling and return the list of parameter samples."""
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.set_default_dtype(torch.float64)

    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

    net = net.to(device, dtype=torch.float64)
    params_init = hamiltorch.util.flatten(net).to(device).clone()

    tau_list = torch.tensor(
        [cfg.tau_prior for _ in net.parameters()],
        dtype=torch.float64,
        device=device,
    )

    x_t = torch.tensor(X_train, dtype=torch.float64, device=device)
    y_t = torch.tensor(y_train, dtype=torch.float64, device=device)
    if x_t.ndim == 1:
        x_t = x_t.unsqueeze(-1)
    if y_t.ndim == 1:
        y_t = y_t.unsqueeze(-1)

    params_hmc = hamiltorch.sample_model(
        net,
        x_t,
        y_t,
        model_loss="regression",
        params_init=params_init,
        num_samples=cfg.num_samples,
        step_size=cfg.step_size,
        num_steps_per_sample=cfg.num_steps_per_sample,
        tau_out=cfg.tau_out,
        tau_list=tau_list,
        verbose=True,
        debug=1,
        store_on_GPU=(device.type == "cuda"),
    )
    return params_hmc


def predict_hmc(
    net: HMCNet,
    X: np.ndarray,
    y: np.ndarray | None,
    params_hmc: list[torch.Tensor],
    cfg: HMCConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Posterior predictive mean and std using HMC samples (after burn-in)."""
    device = params_hmc[0].device
    dtype = params_hmc[0].dtype

    x_t = torch.tensor(X, dtype=dtype, device=device)
    if x_t.ndim == 1:
        x_t = x_t.unsqueeze(-1)

    y_t = None
    if y is not None:
        y_t = torch.tensor(y, dtype=dtype, device=device)
        if y_t.ndim == 1:
            y_t = y_t.unsqueeze(-1)

    tau_list = torch.tensor(
        [cfg.tau_prior for _ in net.parameters()],
        dtype=dtype,
        device=device,
    )

    pred_list, _ = hamiltorch.predict_model(
        net,
        x=x_t,
        y=y_t,
        model_loss="regression",
        samples=params_hmc[:],
        tau_out=cfg.tau_out,
        tau_list=tau_list,
    )

    burn = int(cfg.burn_frac * cfg.num_samples)
    pred_eff = pred_list[burn:].detach().cpu()
    if pred_eff.ndim == 3:
        pred_eff = pred_eff.squeeze(-1)

    mu = pred_eff.mean(0).numpy()
    sd = pred_eff.std(0).numpy()
    return mu, sd
