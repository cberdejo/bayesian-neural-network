"""Suggestions for Optuna models with optional user-defined search spaces."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

import optuna

DEFAULT_SEARCH_SPACES: dict[str, dict[str, Any]] = {
    "MC Dropout": {
        "n_hidden": {"low": 1, "high": 3, "step": 1},
        "units": {"low": 16, "high": 256, "step": 16},
        "dropout_p": {"low": 0.05, "high": 0.5},
        "epochs": {"low": 100, "high": 500, "step": 50},
        "batch_size": {"choices": [16, 32, 64, 128]},
        "lr": {"low": 5e-4, "high": 5e-2, "log": True},
        "mc_samples": {"choices": [50, 100, 200]},
    },
    "VI-BB": {
        "n_hidden": {"low": 1, "high": 3, "step": 1},
        "units": {"low": 16, "high": 128, "step": 16},
        "activation": {"choices": ["tanh", "relu"]},
        "prior_sigma_1": {"low": 0.3, "high": 5.0},
        "prior_sigma_2": {"low": 0.001, "high": 0.5, "log": True},
        "prior_pi": {"low": 0.1, "high": 0.9},
        "kl_weight": {"low": 1e-5, "high": 0.1, "log": True},
        "noise_std": {"low": 0.05, "high": 5.0, "log": True},
        "lr": {"low": 1e-4, "high": 1e-2, "log": True},
        "epochs": {"low": 300, "high": 1500, "step": 100},
        "mc_samples": {"choices": [50, 100, 200]},
    },
    "PBP": {
        "n_hidden": {"low": 1, "high": 3, "step": 1},
        "units": {"low": 10, "high": 100, "step": 10},
        "n_epochs": {"low": 20, "high": 150, "step": 10},
    },
    "HMC": {
        "n_hidden": {"low": 1, "high": 2, "step": 1},
        "units": {"low": 10, "high": 50, "step": 10},
        "step_size": {"low": 1e-4, "high": 1e-2, "log": True},
        "num_samples": {"low": 100, "high": 400, "step": 50},
        "num_steps_per_sample": {"low": 5, "high": 30, "step": 1},
        "tau_out": {"low": 1.0, "high": 500.0, "log": True},
        "tau_prior": {"low": 0.01, "high": 20.0, "log": True},
        "burn_frac": {"low": 0.2, "high": 0.5},
    },
    "ABC-SS": {
        "n_hidden": {"low": 1, "high": 3, "step": 1},
        "units": {"low": 8, "high": 64, "step": 8},
        "activation": {"choices": ["tanh", "relu", "sigmoid"]},
        "n_samples": {"low": 2000, "high": 8000, "step": 500},
        "sim_levels": {"low": 3, "high": 6, "step": 1},
        "p0": {"low": 0.1, "high": 0.3},
        "initial_std": {"low": 0.2, "high": 2.0},
        "prior_half": {"low": 0.5, "high": 4.0},
        "n_best_frac": {"low": 0.05, "high": 0.2},
    },
}


def _merged_space(model_name: str, custom_space: dict[str, Any] | None) -> dict[str, Any]:
    merged = deepcopy(DEFAULT_SEARCH_SPACES[model_name])
    if custom_space:
        for key, value in custom_space.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key].update(value)
            else:
                merged[key] = value
    return merged


def _suggest_int(
    trial: optuna.Trial,
    name: str,
    spec: dict[str, Any],
) -> int:
    return trial.suggest_int(
        name,
        int(spec["low"]),
        int(spec["high"]),
        step=int(spec.get("step", 1)),
    )


def _suggest_float(
    trial: optuna.Trial,
    name: str,
    spec: dict[str, Any],
) -> float:
    return trial.suggest_float(
        name,
        float(spec["low"]),
        float(spec["high"]),
        log=bool(spec.get("log", False)),
    )


def _suggest_categorical(
    trial: optuna.Trial,
    name: str,
    spec: dict[str, Any],
) -> Any:
    return trial.suggest_categorical(name, spec["choices"])


def _suggest_mc_dropout(
    trial: optuna.Trial,
    input_dim: int,
    custom_space: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del input_dim
    space = _merged_space("MC Dropout", custom_space)
    n = _suggest_int(trial, "n_hidden", space["n_hidden"])
    hidden = [_suggest_int(trial, f"units_{i}", space["units"]) for i in range(n)]
    return {
        "hidden_layers": ",".join(map(str, hidden)),
        "dropout_p": _suggest_float(trial, "dropout_p", space["dropout_p"]),
        "epochs": _suggest_int(trial, "epochs", space["epochs"]),
        "batch_size": _suggest_categorical(trial, "batch_size", space["batch_size"]),
        "lr": _suggest_float(trial, "lr", space["lr"]),
        "mc_samples": _suggest_categorical(trial, "mc_samples", space["mc_samples"]),
        "seed": 42,
    }


def _suggest_vi_bb(
    trial: optuna.Trial,
    input_dim: int,
    custom_space: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del input_dim
    space = _merged_space("VI-BB", custom_space)
    n = _suggest_int(trial, "n_hidden", space["n_hidden"])
    hidden = [_suggest_int(trial, f"units_{i}", space["units"]) for i in range(n)]
    return {
        "hidden_layers": ",".join(map(str, hidden)),
        "activation": _suggest_categorical(trial, "activation", space["activation"]),
        "prior_sigma_1": _suggest_float(trial, "prior_sigma_1", space["prior_sigma_1"]),
        "prior_sigma_2": _suggest_float(trial, "prior_sigma_2", space["prior_sigma_2"]),
        "prior_pi": _suggest_float(trial, "prior_pi", space["prior_pi"]),
        "kl_weight": _suggest_float(trial, "kl_weight", space["kl_weight"]),
        "noise_std": _suggest_float(trial, "noise_std", space["noise_std"]),
        "lr": _suggest_float(trial, "lr", space["lr"]),
        "epochs": _suggest_int(trial, "epochs", space["epochs"]),
        "mc_samples": _suggest_categorical(trial, "mc_samples", space["mc_samples"]),
        "seed": 42,
    }


def _suggest_pbp(
    trial: optuna.Trial,
    input_dim: int,
    custom_space: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del input_dim
    space = _merged_space("PBP", custom_space)
    n = _suggest_int(trial, "n_hidden", space["n_hidden"])
    hidden = [_suggest_int(trial, f"units_{i}", space["units"]) for i in range(n)]
    return {
        "hidden_layers": ",".join(map(str, hidden)),
        "n_epochs": _suggest_int(trial, "n_epochs", space["n_epochs"]),
        "seed": 42,
    }


def _suggest_hmc(
    trial: optuna.Trial,
    input_dim: int,
    custom_space: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del input_dim
    space = _merged_space("HMC", custom_space)
    n = _suggest_int(trial, "n_hidden", space["n_hidden"])
    hidden = [_suggest_int(trial, f"units_{i}", space["units"]) for i in range(n)]
    return {
        "hidden_layers": ",".join(map(str, hidden)),
        "step_size": _suggest_float(trial, "step_size", space["step_size"]),
        "num_samples": _suggest_int(trial, "num_samples", space["num_samples"]),
        "num_steps_per_sample": _suggest_int(
            trial, "num_steps_per_sample", space["num_steps_per_sample"]
        ),
        "tau_out": _suggest_float(trial, "tau_out", space["tau_out"]),
        "tau_prior": _suggest_float(trial, "tau_prior", space["tau_prior"]),
        "burn_frac": _suggest_float(trial, "burn_frac", space["burn_frac"]),
        "seed": 42,
    }


def _suggest_abc_ss(
    trial: optuna.Trial,
    input_dim: int,
    custom_space: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del input_dim
    space = _merged_space("ABC-SS", custom_space)
    n = _suggest_int(trial, "n_hidden", space["n_hidden"])
    hidden = [_suggest_int(trial, f"units_{i}", space["units"]) for i in range(n)]

    n_samples = _suggest_int(trial, "n_samples", space["n_samples"])
    prior_half = _suggest_float(trial, "prior_half", space["prior_half"])
    n_best_frac = _suggest_float(trial, "n_best_frac", space["n_best_frac"])
    n_best = max(50, int(n_samples * n_best_frac))

    return {
        "hidden_layers": ",".join(map(str, hidden)),
        "activation": _suggest_categorical(trial, "activation", space["activation"]),
        "n_samples": n_samples,
        "sim_levels": _suggest_int(trial, "sim_levels", space["sim_levels"]),
        "p0": _suggest_float(trial, "p0", space["p0"]),
        "initial_std": _suggest_float(trial, "initial_std", space["initial_std"]),
        "prior_low": -prior_half,
        "prior_high": prior_half,
        "n_best": n_best,
        "seed": 42,
    }


def get_suggester(
    model_name: str,
    custom_space: dict[str, Any] | None = None,
) -> Callable[[optuna.Trial, int], dict[str, Any]]:
    suggesters: dict[str, Callable[..., dict[str, Any]]] = {
        "MC Dropout": _suggest_mc_dropout,
        "VI-BB": _suggest_vi_bb,
        "PBP": _suggest_pbp,
        "HMC": _suggest_hmc,
        "ABC-SS": _suggest_abc_ss,
    }
    base = suggesters[model_name]
    return lambda trial, input_dim: base(trial, input_dim, custom_space)


SUGGESTERS: dict[str, Callable[[optuna.Trial, int], dict[str, Any]]] = {
    model_name: get_suggester(model_name) for model_name in DEFAULT_SEARCH_SPACES
}
 