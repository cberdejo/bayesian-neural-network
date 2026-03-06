from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
import tqdm
from torch import nn
from torch.nn import functional as F

from .config import VIBBConfig


class DenseVariational(nn.Module):
    """Fully-connected layer with variational Gaussian posterior and mixture prior."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kl_weight: float,
        activation: str | None = None,
        prior_sigma_1: float = 1.5,
        prior_sigma_2: float = 0.1,
        prior_pi: float = 0.5,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kl_weight = float(kl_weight)

        activations = {"tanh": nn.Tanh, "relu": nn.ReLU}
        if activation is None:
            self.act = nn.Identity()
        elif activation.lower() in activations:
            self.act = activations[activation.lower()]()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.prior_sigma_1 = float(prior_sigma_1)
        self.prior_sigma_2 = float(prior_sigma_2)
        self.prior_pi_1 = float(prior_pi)
        self.prior_pi_2 = 1.0 - self.prior_pi_1

        init_sigma = math.sqrt(
            self.prior_pi_1 * self.prior_sigma_1**2
            + self.prior_pi_2 * self.prior_sigma_2**2
        )

        self.kernel_mu = nn.Parameter(
            torch.randn(in_features, out_features) * init_sigma
        )
        self.kernel_rho = nn.Parameter(torch.zeros(in_features, out_features))
        self.bias_mu = nn.Parameter(torch.randn(out_features) * init_sigma)
        self.bias_rho = nn.Parameter(torch.zeros(out_features))

    def _log_mixture_prior(self, z: torch.Tensor) -> torch.Tensor:
        log_p1 = (
            math.log(self.prior_pi_1)
            - 0.5 * math.log(2.0 * math.pi)
            - math.log(self.prior_sigma_1)
            - 0.5 * (z / self.prior_sigma_1) ** 2
        )
        log_p2 = (
            math.log(self.prior_pi_2)
            - 0.5 * math.log(2.0 * math.pi)
            - math.log(self.prior_sigma_2)
            - 0.5 * (z / self.prior_sigma_2) ** 2
        )
        return torch.logaddexp(log_p1, log_p2)

    @staticmethod
    def _log_gaussian(
        x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        return (
            -0.5 * math.log(2.0 * math.pi)
            - torch.log(sigma)
            - 0.5 * ((x - mu) / sigma) ** 2
        )

    def _sample_params_and_kl(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k_sigma = F.softplus(self.kernel_rho)
        b_sigma = F.softplus(self.bias_rho)

        W = self.kernel_mu + k_sigma * torch.randn_like(self.kernel_mu)
        b = self.bias_mu + b_sigma * torch.randn_like(self.bias_mu)

        log_q = self._log_gaussian(W, self.kernel_mu, k_sigma).sum() + self._log_gaussian(
            b, self.bias_mu, b_sigma
        ).sum()
        log_p = self._log_mixture_prior(W).sum() + self._log_mixture_prior(b).sum()

        kl = self.kl_weight * (log_q - log_p)
        return W, b, kl

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        W, b, kl = self._sample_params_and_kl()
        return self.act(x @ W + b), kl


def nll_gaussian(
    y_true: torch.Tensor, y_pred: torch.Tensor, sigma: float
) -> torch.Tensor:
    """Gaussian negative log-likelihood (summed over all data points)."""
    return torch.sum(
        0.5 * ((y_true - y_pred) / sigma) ** 2
        + math.log(sigma)
        + 0.5 * math.log(2.0 * math.pi)
    )


class BayesianMLP(nn.Module):
    """Multi-layer Bayesian MLP with variational inference (Bayes by Backprop)."""

    def __init__(
        self,
        layer_sizes: Sequence[int],
        kl_weight: float = 1.0,
        activation: str = "tanh",
        prior_sigma_1: float = 1.5,
        prior_sigma_2: float = 0.1,
        prior_pi: float = 0.5,
    ):
        super().__init__()
        prior_kw = dict(
            prior_sigma_1=prior_sigma_1,
            prior_sigma_2=prior_sigma_2,
            prior_pi=prior_pi,
        )
        layers: list[DenseVariational] = []
        for i in range(len(layer_sizes) - 1):
            is_last = i == len(layer_sizes) - 2
            act = None if is_last else activation
            layers.append(
                DenseVariational(
                    layer_sizes[i], layer_sizes[i + 1], kl_weight, activation=act, **prior_kw
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        kl_total = torch.tensor(0.0, device=x.device)
        for layer in self.layers:
            x, kl = layer(x)
            kl_total = kl_total + kl
        return x, kl_total

    @classmethod
    def from_config(cls, cfg: VIBBConfig) -> "BayesianMLP":
        return cls(
            layer_sizes=cfg.layer_sizes,
            kl_weight=cfg.kl_weight,
            activation=cfg.activation,
            prior_sigma_1=cfg.prior_sigma_1,
            prior_sigma_2=cfg.prior_sigma_2,
            prior_pi=cfg.prior_pi,
        )

    def train_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cfg: VIBBConfig,
    ) -> list[float]:
        device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.to(device)

        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        y_t = torch.tensor(y, dtype=torch.float32, device=device)

        optimizer = torch.optim.Adam(self.parameters(), lr=cfg.lr)

        mse_history: list[float] = []
        self.train()
        for _ in tqdm.tqdm(range(cfg.epochs), desc="Training VI-BB"):
            optimizer.zero_grad()
            y_pred, kl = self(X_t)
            nll = nll_gaussian(y_t, y_pred, sigma=cfg.noise_std)
            loss = nll + kl
            loss.backward()
            optimizer.step()
            mse_history.append(torch.mean((y_pred - y_t) ** 2).item())
        return mse_history

    def predict(
        self,
        X: np.ndarray,
        cfg: VIBBConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, std) via Monte Carlo sampling from the posterior."""
        device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        X_t = torch.tensor(X, dtype=torch.float32, device=device)

        self.eval()
        samples: list[np.ndarray] = []
        with torch.no_grad():
            for _ in tqdm.tqdm(range(cfg.mc_samples), desc="MC sampling"):
                y_s, _ = self(X_t)
                samples.append(y_s.cpu().numpy())

        y_preds = np.concatenate(samples, axis=1)
        if y_preds.ndim == 3:
            y_preds = y_preds.squeeze(-1)
        return y_preds.mean(axis=1), y_preds.std(axis=1)
