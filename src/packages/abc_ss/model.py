from __future__ import annotations

import sys
from typing import Callable, Sequence

import numpy as np
from scipy import stats

from .config import ABCSSConfig

_ACTIVATIONS: dict[str, tuple[Callable, Callable]] = {
    "sigmoid": (
        lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500))),
        lambda x: x * (1 - x),
    ),
    "tanh": (
        lambda x: np.tanh(x),
        lambda x: 1 - np.tanh(x) ** 2,
    ),
    "relu": (
        lambda x: np.maximum(x, 0),
        lambda x: (x > 0).astype(float),
    ),
}


def _count_weights(neurons: Sequence[int]) -> tuple[int, int]:
    nW = sum(neurons[i] * neurons[i + 1] for i in range(len(neurons) - 1))
    nb = sum(neurons[i + 1] for i in range(len(neurons) - 1))
    return nW, nb


def _vec_to_matrices(W_vec: np.ndarray, neurons: Sequence[int]) -> list[np.ndarray]:
    matrices = []
    ref = 0
    for i in range(len(neurons) - 1):
        size = neurons[i] * neurons[i + 1]
        matrices.append(W_vec[ref: ref + size].reshape(neurons[i], neurons[i + 1]))
        ref += size
    return matrices


def _vec_to_biases(b_vec: np.ndarray, neurons: Sequence[int]) -> list[np.ndarray]:
    biases = []
    ref = 0
    for i in range(len(neurons) - 1):
        size = neurons[i + 1]
        biases.append(b_vec[ref: ref + size].reshape(1, size))
        ref += size
    return biases


def _forward_pass(
    X: np.ndarray,
    W_mats: list[np.ndarray],
    b_mats: list[np.ndarray],
    act_funcs: list[tuple[Callable, Callable]],
) -> np.ndarray:
    output = X
    for idx in range(len(W_mats)):
        z = output @ W_mats[idx] + b_mats[idx]
        output = act_funcs[idx][0](z)
    return output


def _mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2))


class ABCSubSim:
    """Approximate Bayesian Computation by Subset Simulation for BNNs."""

    def __init__(self, cfg: ABCSSConfig):
        self.cfg = cfg
        self.neurons = cfg.layer_sizes
        self.nW, self.nb = _count_weights(self.neurons)
        self.n_params = self.nW + self.nb

        self.act_funcs = [_ACTIVATIONS[a] for a in cfg.activations]
        if len(self.act_funcs) != len(self.neurons) - 1:
            raise ValueError(
                f"Expected {len(self.neurons) - 1} activations, got {len(self.act_funcs)}."
            )

        self.samples: np.ndarray | None = None
        self.intermediate_samples: np.ndarray | None = None
        self.epsilons: list[float] = []

    def _evaluate_sample(self, sample: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        W = _vec_to_matrices(sample[: self.nW], self.neurons)
        b = _vec_to_biases(sample[self.nW: self.nW + self.nb], self.neurons)
        y_pred = _forward_pass(X, W, b, self.act_funcs)
        return _mse(y_pred, y)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ABCSubSim":
        cfg = self.cfg
        N = cfg.n_samples
        sim_levels = cfg.sim_levels
        p0 = cfg.p0
        new_std = cfg.initial_std if cfg.initial_std is not None else (sim_levels + 1) * 0.1

        if cfg.seed is not None:
            np.random.seed(cfg.seed)

        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_total = self.nW + self.nb + 1
        all_levels = np.zeros((sim_levels + 1, N, n_total))

        samples = np.zeros((N, n_total))
        for i in range(self.n_params):
            samples[:, i] = stats.truncnorm.rvs(
                cfg.prior_low, cfg.prior_high, loc=0, scale=1, size=N
            )

        for i in range(N):
            samples[i, -1] = self._evaluate_sample(samples[i], X, y)
        all_levels[0] = samples

        self.epsilons = []
        for m in range(sim_levels):
            sorted_samples = samples[samples[:, -1].argsort()]
            per_ind = int(np.rint(p0 * N))
            intr_eps = sorted_samples[per_ind, -1]
            self.epsilons.append(float(intr_eps))
            prop_std = new_std - (m + 1) * 0.1

            seeds = sorted_samples[:per_ind]
            n_new = N - per_ind
            new_samples = np.zeros((n_new, n_total))
            idx = 0

            chains_per_seed = int(np.rint(1 / p0)) - 1
            for k in range(per_ind):
                prev = seeds[k].copy()
                for _ in range(chains_per_seed):
                    candidate = prev.copy()
                    candidate[:self.n_params] = (
                        prop_std * np.random.randn(self.n_params) + prev[:self.n_params]
                    )
                    e = self._evaluate_sample(candidate, X, y)
                    candidate[-1] = e
                    if e <= intr_eps:
                        new_samples[idx] = candidate
                        prev = candidate.copy()
                    else:
                        new_samples[idx] = prev.copy()
                    idx += 1

            samples = np.concatenate([seeds, new_samples], axis=0)
            all_levels[m + 1] = samples

            sys.stdout.write(f"Level {m + 1}/{sim_levels} — ε={intr_eps:.6f}\n")
            sys.stdout.flush()

        self.samples = samples
        self.intermediate_samples = all_levels
        return self

    def predict(
        self, X: np.ndarray, n_best: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict using the best posterior samples. Returns (mean, std)."""
        if self.samples is None:
            raise RuntimeError("Call fit() first.")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        sorted_samples = self.samples[self.samples[:, -1].argsort()]
        if n_best is not None:
            sorted_samples = sorted_samples[:n_best]

        preds: list[np.ndarray] = []
        for s in sorted_samples:
            W = _vec_to_matrices(s[: self.nW], self.neurons)
            b = _vec_to_biases(s[self.nW: self.nW + self.nb], self.neurons)
            preds.append(_forward_pass(X, W, b, self.act_funcs))

        preds_arr = np.array(preds).squeeze(-1)
        return preds_arr.mean(axis=0), preds_arr.std(axis=0)
