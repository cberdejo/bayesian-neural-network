"""
hpo.py – Hyperparameter optimisation with Optuna for Bayesian-NN Streamlit app.

The UI controls (n_trials, alpha, objectives) live in app.py.
This module runs the study, renders the results, and — in single-objective
mode — adds an inference tab inside the diagnostics section.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np
import optuna
import polars as pl
import streamlit as st

from inference_results import plot_inference  
from metrics.compute_metrics import compute_metrics
from suggestions_optuna_models import get_suggester

# ──────────────────────────────────────────────────────────────────────────────
# Reconstruct run_model-compatible params from a frozen Optuna trial
# ──────────────────────────────────────────────────────────────────────────────

def _params_from_trial(trial: optuna.FrozenTrial, model_name: str | None = None) -> dict[str, Any]:
    """
    Optuna stores architecture as n_hidden + units_0, units_1, …
    run_model() expects a single "hidden_layers" string like "32,16".
    
    For ABC-SS, also reconstructs derived parameters:
    - prior_low, prior_high from prior_half
    - n_best from n_samples and n_best_frac
    """
    p = dict(trial.params)
    n_hidden = p.pop("n_hidden", None)
    if n_hidden is not None:
        hidden = [p.pop(f"units_{i}") for i in range(int(n_hidden))]
        p["hidden_layers"] = ",".join(map(str, hidden))
    
    # Reconstruct ABC-SS derived parameters
    if model_name == "ABC-SS":
        if "prior_half" in p and "prior_low" not in p:
            prior_half = p.pop("prior_half")
            p["prior_low"] = -prior_half
            p["prior_high"] = prior_half
        if "n_samples" in p and "n_best_frac" in p and "n_best" not in p:
            n_samples = p["n_samples"]
            n_best_frac = p["n_best_frac"]
            p["n_best"] = max(50, int(n_samples * n_best_frac))
    
    p.setdefault("seed", 42)
    return p


# ──────────────────────────────────────────────────────────────────────────────
# Objective factory
# ──────────────────────────────────────────────────────────────────────────────

def _build_objective(
    model_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    objectives: list[str],
    alpha: float,
    run_model_fn: Callable,
    search_space: dict[str, Any] | None = None,
    scaler_y=None,
) -> Callable:
    suggest_fn = get_suggester(model_name, search_space)

    def objective(trial: optuna.Trial):
        params = suggest_fn(trial, X_train.shape[1])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y_pred, y_std, _ = run_model_fn(model_name, X_train, X_test, y_train, params)
        except Exception as exc:
            raise optuna.exceptions.TrialPruned(f"Trial failed: {exc}")

        if scaler_y is not None:
            y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            y_std  = y_std * float(scaler_y.scale_[0])

        m = compute_metrics(y_test, y_pred, y_std, alpha=alpha)
        for k, v in m.items():
            trial.set_user_attr(k, round(float(v), 6))

        picp_loss = (m["PICP"] - (1.0 - alpha)) ** 2
        obj_map = {
            "RMSE":      m["RMSE"],
            "MAE":       m["MAE"],
            "PICP loss": picp_loss,
            "MPIW":      m["MPIW"],
            "NLL":       m["NLL"],
            "Winkler":   m["Winkler"],
        }
        values = [obj_map[o] for o in objectives]
        return values[0] if len(values) == 1 else tuple(values)

    return objective


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def render_hpo_section(
    model_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    run_model_fn: Callable,
    *,
    n_trials: int,
    alpha: float,
    objectives: list[str],
    search_space: dict[str, Any] | None = None,
    scaler_y=None,
) -> tuple[np.ndarray, np.ndarray, dict] | None:
    """
    Runs an Optuna study and renders results into the Streamlit page.

    Returns (y_pred, y_std, cfg_dict) for the best trial in single-objective
    mode (already inverse-transformed), or None in multi-objective mode.
    """
    return _run_study(
        model_name=model_name,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        run_model_fn=run_model_fn,
        objectives=objectives,
        n_trials=n_trials,
        alpha=alpha,
        search_space=search_space,
        scaler_y=scaler_y,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Study runner
# ──────────────────────────────────────────────────────────────────────────────

def _run_study(
    model_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    run_model_fn: Callable,
    objectives: list[str],
    n_trials: int,
    alpha: float,
    search_space: dict[str, Any] | None = None,
    scaler_y=None,
) -> tuple[np.ndarray, np.ndarray, dict] | None:
    
    multi   = len(objectives) > 1
    sampler = (
        optuna.samplers.NSGAIISampler(seed=42) if multi
        else optuna.samplers.TPESampler(seed=42)
    )
    study = optuna.create_study(
        directions=["minimize"] * len(objectives),
        sampler=sampler,
    )
    obj_fn = _build_objective(
        model_name, X_train, X_test, y_train, y_test,
        objectives, alpha, run_model_fn, search_space, scaler_y,
    )

    progress_bar  = st.progress(0, text="Starting…")
    status_txt    = st.empty()
    log_container = st.empty()
    log_lines: list[str] = []

    def _cb(study: optuna.Study, trial: optuna.FrozenTrial) -> None:
        done   = trial.number + 1
        pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
        progress_bar.progress(int(done / n_trials * 100), text=f"Trial {done}/{n_trials}")
        status_txt.caption(f"Completed: {done - pruned}  |  Pruned: {pruned}")
        if trial.state == optuna.trial.TrialState.COMPLETE:
            vals_str = "  |  ".join(
                f"{o}={v:.4f}" for o, v in zip(objectives, _as_list(trial.values))
            )
            log_lines.append(f"✅ {done:>3}: {vals_str}")
        else:
            log_lines.append(f"✂️  {done:>3}: pruned")
        log_container.code("\n".join(log_lines[-14:]))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with st.spinner("Optimising…"):
        study.optimize(obj_fn, n_trials=n_trials, callbacks=[_cb], show_progress_bar=False)

    progress_bar.progress(100, text="Done ✔")
    st.success(f"Finished — {n_trials} trials attempted.")

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        st.error("All trials were pruned. Relax the search space or increase n_trials.")
        return None

    # ── Full results table ────────────────────────────────────────────────────
    st.subheader("📊 All completed trials")
    rows = []
    for t in completed:
        row = {"trial": t.number}
        for o, v in zip(objectives, _as_list(t.values)):
            row[f"▼ {o}"] = round(v, 5)
        row.update(t.user_attrs)
        row.update({f"param_{k}": v for k, v in t.params.items()})
        rows.append(row)
    df_all = pl.DataFrame(rows).sort(f"▼ {objectives[0]}")
    st.dataframe(df_all, use_container_width=True)

    # ── Best / Pareto ─────────────────────────────────────────────────────────
    best_result: tuple[np.ndarray, np.ndarray, dict] | None = None

    if multi:
        best_trials = study.best_trials
        st.subheader(f"🏅 Pareto front — {len(best_trials)} non-dominated trials")
        _show_pareto(best_trials, objectives)

        # Pick a single representative trial from the Pareto front.
        # Strategy: minimize normalized L2 distance to the ideal point.
        compromise_trial = _select_compromise_trial(best_trials)
        st.markdown("**Compromise trial (best overall)**")
        _show_single_best(compromise_trial, objectives, model_name)

        # Re-run representative trial to enable inference diagnostics in multi-objective mode.
        best_params = _params_from_trial(compromise_trial, model_name)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y_pred, y_std, cfg_dict = run_model_fn(
                    model_name, X_train, X_test, y_train, best_params
                )
            if scaler_y is not None:
                y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                y_std  = y_std * float(scaler_y.scale_[0])
            best_result = (y_pred, y_std, cfg_dict)
        except Exception as exc:
            st.warning(f"Could not re-run compromise trial for inference plot: {exc}")
    else:
        best_trial = study.best_trial
        st.subheader("🏅 Best trial")
        _show_single_best(best_trial, objectives, model_name)

        # Re-run with reconstructed params to get predictions for the inference tab
        best_params = _params_from_trial(best_trial, model_name)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y_pred, y_std, cfg_dict = run_model_fn(
                    model_name, X_train, X_test, y_train, best_params
                )
            if scaler_y is not None:
                y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                y_std  = y_std * float(scaler_y.scale_[0])
            best_result = (y_pred, y_std, cfg_dict)
        except Exception as exc:
            st.warning(f"Could not re-run best trial for inference plot: {exc}")

    # ── Diagnostics tabs ──────────────────────────────────────────────────────
    st.subheader("📈 Diagnostics")
    _render_diagnostics(study, objectives, multi, y_test, best_result, alpha)

    return best_result


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostics tabs
# ──────────────────────────────────────────────────────────────────────────────

def _render_diagnostics(
    study: optuna.Study,
    objectives: list[str],
    multi: bool,
    y_test: np.ndarray,
    best_result: tuple[np.ndarray, np.ndarray, dict] | None,
    alpha: float,
) -> None:
    # Inference tab is only shown in single-objective mode where a best model exists
    tab_names = ["Optimisation history", "Param importances", "Parallel coordinates"]
    if best_result is not None:
        tab_names.append("🔍 Inference (best trial)")

    tabs = st.tabs(tab_names)

    # ── Optuna visualisations ─────────────────────────────────────────────────
    try:
        from optuna.visualization import (
            plot_optimization_history,
            plot_param_importances,
            plot_parallel_coordinate,
        )

        with tabs[0]:
            if multi:
                cols = st.columns(min(len(objectives), 3))
                for i, (obj, col) in enumerate(zip(objectives, cols)):
                    fig = plot_optimization_history(
                        study,
                        target=lambda t, i=i: _as_list(t.values)[i],
                        target_name=obj,
                    )
                    col.plotly_chart(fig, use_container_width=True)
            else:
                st.plotly_chart(plot_optimization_history(study), use_container_width=True)

        with tabs[1]:
            fig = plot_param_importances(
                study,
                target=(lambda t: _as_list(t.values)[0]) if multi else None,
                target_name=objectives[0],
            )
            st.plotly_chart(fig, use_container_width=True)

        with tabs[2]:
            fig = plot_parallel_coordinate(
                study,
                target=(lambda t: _as_list(t.values)[0]) if multi else None,
                target_name=objectives[0],
            )
            st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        for tab in tabs[:3]:
            with tab:
                st.info("Install `plotly` for interactive plots:  `pip install plotly`")
    except Exception as exc:
        for tab in tabs[:3]:
            with tab:
                st.warning(f"Could not render Optuna plots: {exc}")

    # ── Inference tab ─────────────────────────────────
    if best_result is not None:
        y_pred, y_std, cfg_dict = best_result
        metrics = compute_metrics(y_test, y_pred, y_std, alpha=alpha)
        with tabs[3]:
            st.pyplot(plot_inference(y_test, y_pred, y_std, alpha=alpha), use_container_width=True)

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("RMSE",    f"{metrics['RMSE']:.4f}")
            m2.metric("PICP",    f"{metrics['PICP']:.4f}")
            m3.metric("MPIW",    f"{metrics['MPIW']:.4f}")
            m4.metric("NLL",     f"{metrics['NLL']:.4f}")
            m5.metric("Winkler", f"{metrics['Winkler']:.4f}")

            with st.expander("Best hyperparameters"):
                st.json(cfg_dict)


# ──────────────────────────────────────────────────────────────────────────────
# Display helpers
# ──────────────────────────────────────────────────────────────────────────────

def _as_list(values) -> list[float]:
    if values is None:
        return []
    try:
        return list(values)
    except TypeError:
        return [float(values)]


def _show_single_best(trial: optuna.FrozenTrial, objectives: list[str], model_name: str | None = None) -> None:
    vals = _as_list(trial.values)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Objective values**")
        st.table(pl.DataFrame({"Objective": objectives, "Value": [round(v, 5) for v in vals]}))
        st.markdown("**All metrics**")
        st.table(pl.DataFrame({
            "Metric": list(trial.user_attrs.keys()),
            "Value":  [round(v, 5) for v in trial.user_attrs.values()],
        }))
    with c2:
        st.markdown("**Best hyperparameters**")
        st.json(_params_from_trial(trial, model_name))


def _select_compromise_trial(best_trials: list[optuna.FrozenTrial]) -> optuna.FrozenTrial:
    """
    Select a single Pareto trial that best balances all objectives.
    All objectives are normalized to [0, 1] and ranked by L2 distance to ideal.
    """
    values_matrix = np.asarray([_as_list(t.values) for t in best_trials], dtype=float)
    mins = values_matrix.min(axis=0)
    maxs = values_matrix.max(axis=0)
    spans = np.where(maxs > mins, maxs - mins, 1.0)
    normalized = (values_matrix - mins) / spans
    distances = np.linalg.norm(normalized, axis=1)
    best_idx = int(np.argmin(distances))
    return best_trials[best_idx]


def _show_pareto(best_trials: list[optuna.FrozenTrial], objectives: list[str]) -> None:
    import matplotlib.pyplot as plt

    rows = []
    for t in best_trials:
        row = {"trial": t.number}
        row.update({o: round(v, 5) for o, v in zip(objectives, _as_list(t.values))})
        row.update(t.user_attrs)
        rows.append(row)
    st.dataframe(pl.DataFrame(rows), use_container_width=True)

    if len(objectives) == 2:
        fig, ax = plt.subplots(figsize=(6, 4))
        xs = [_as_list(t.values)[0] for t in best_trials]
        ys = [_as_list(t.values)[1] for t in best_trials]
        ax.scatter(xs, ys, c="steelblue", edgecolors="k", linewidths=0.4, s=70)
        for t, x, y in zip(best_trials, xs, ys):
            ax.annotate(str(t.number), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax.set_xlabel(objectives[0])
        ax.set_ylabel(objectives[1])
        ax.set_title("Pareto front")
        fig.tight_layout()
        st.pyplot(fig)