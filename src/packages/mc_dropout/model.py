from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .config import MCDropoutConfig


def _gaussian_nll(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    y_true = y_true.reshape(-1)
    mu = y_pred[:, 0]
    log_var = y_pred[:, 1]
    loss = (log_var + (y_true - mu).pow(2) / torch.exp(log_var)) / 2.0
    return loss.mean()


class MCDropoutNet(nn.Module):
    """MLP with dropout for Monte Carlo Dropout inference.

    The last layer outputs 2 values: (mu, log_var) to model heteroscedastic
    aleatoric uncertainty alongside the epistemic uncertainty from dropout.
    """

    def __init__(
        self,
        layer_sizes: Sequence[int],
        dropout_p: float = 0.5,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                if i % 2 == 0:
                    layers.append(nn.ReLU())
                else:
                    layers.append(nn.Tanh())
                layers.append(nn.Dropout(p=dropout_p))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @classmethod
    def from_config(cls, cfg: MCDropoutConfig) -> "MCDropoutNet":
        return cls(layer_sizes=cfg.layer_sizes, dropout_p=cfg.dropout_p)


def train_mc_dropout(
    model: MCDropoutNet,
    X: np.ndarray,
    y: np.ndarray,
    cfg: MCDropoutConfig,
) -> MCDropoutNet:
    """Train the MC-Dropout model and return it."""
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

    model = model.to(device)
    x_t = torch.as_tensor(X, dtype=torch.float32).reshape(-1, cfg.layer_sizes[0])
    y_t = torch.as_tensor(y, dtype=torch.float32).reshape(-1)
    ds = TensorDataset(x_t, y_t)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    model.train()
    for _ in range(cfg.epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = _gaussian_nll(yb, pred)
            optim.zero_grad()
            loss.backward()
            optim.step()
    return model


def predict_mc_dropout(
    model: MCDropoutNet,
    X: np.ndarray,
    cfg: MCDropoutConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict using MC Dropout (keep dropout active). Returns (mean, std)."""
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    x_t = torch.as_tensor(X, dtype=torch.float32, device=device).reshape(-1, cfg.layer_sizes[0])

    mu_list: list[np.ndarray] = []
    logvar_list: list[np.ndarray] = []

    model.train()
    with torch.no_grad():
        for _ in range(cfg.mc_samples):
            y_pred = model(x_t)
            mu_list.append(y_pred[:, 0].cpu().numpy())
            logvar_list.append(y_pred[:, 1].cpu().numpy())

    mu_arr = np.stack(mu_list, axis=0)
    logvar_arr = np.stack(logvar_list, axis=0)
    var_arr = np.exp(logvar_arr)

    y_mean = np.mean(mu_arr, axis=0)
    y_variance = np.mean(var_arr + mu_arr**2, axis=0) - y_mean**2
    y_std = np.sqrt(np.maximum(y_variance, 0.0))
    return y_mean, y_std
