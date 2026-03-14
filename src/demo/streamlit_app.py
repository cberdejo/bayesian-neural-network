from __future__ import annotations

import io
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import polars as pl
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from packages.abc_ss import ABCSSConfig, ABCSubSim
from packages.hmc import HMCConfig, HMCNet, predict_hmc, sample_hmc
from packages.mc_dropout import (
    MCDropoutConfig,
    MCDropoutNet,
    predict_mc_dropout,
    train_mc_dropout,
)
from packages.pbp import PBPConfig, PBP_net
from packages.vi_bb import BayesianMLP, VIBBConfig
from inference_results import render_results  
from metrics.compute_metrics import compute_metrics
from hpo import render_hpo_section
from metric_explanation import show_metric_explanations
DATASET_PATH = PROJECT_ROOT / "dataset" / "Concrete_Data.xls"
MODEL_OPTIONS = ["MC Dropout", "VI-BB", "PBP", "HMC", "ABC-SS"]
NOTEBOOK_URLS = {
    "MC Dropout": (
        "https://github.com/cberdejo/bayesian-neural-network/"
        "blob/main/src/notebooks/mc-dropout/mc_dropout.ipynb"
    ),
    "VI-BB": (
        "https://github.com/cberdejo/bayesian-neural-network/"
        "blob/main/src/notebooks/vi-bb/vi_bb.ipynb"
    ),
    "PBP": (
        "https://github.com/cberdejo/bayesian-neural-network/"
        "blob/main/src/notebooks/pbp/pbp.ipynb"
    ),
    "HMC": (
        "https://github.com/cberdejo/bayesian-neural-network/"
        "blob/main/src/notebooks/hmc/hmc.ipynb"
    ),
    "ABC-SS": (
        "https://github.com/cberdejo/bayesian-neural-network/"
        "blob/main/src/notebooks/abc-ss/abc_ss.ipynb"
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ──────────────────────────────────────────────────────────────────────────────

def parse_hidden_layers(raw: str) -> list[int]:
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("You must enter at least one hidden layer. Example: 64,32")
    hidden = [int(v) for v in values]
    if any(v <= 0 for v in hidden):
        raise ValueError("Hidden layers must be positive integers.")
    return hidden


@st.cache_data(show_spinner=False)
def load_and_clean_dataset(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return _clean_dataset(_read_dataset(path, path.name))


@st.cache_data(show_spinner=False)
def load_and_clean_uploaded_dataset(file_bytes: bytes, file_name: str) -> pl.DataFrame:
    return _clean_dataset(_read_dataset(io.BytesIO(file_bytes), file_name))


def _read_dataset(source: Path | io.BytesIO, file_name: str) -> pl.DataFrame:
    suffix = Path(file_name).suffix.lower()
    if suffix == ".csv":
        return pl.read_csv(source)
    if suffix in {".xls", ".xlsx"}:
        return pl.read_excel(source, engine="calamine")
    raise ValueError("Unsupported format. Use .xls, .xlsx or .csv")


def _clean_dataset(df: pl.DataFrame) -> pl.DataFrame:
    return df.unique(maintain_order=True).drop_nulls()


# ──────────────────────────────────────────────────────────────────────────────
# Model runner
# ──────────────────────────────────────────────────────────────────────────────

def run_model(
    model_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    params: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    input_dim = X_train.shape[1]
    hidden = parse_hidden_layers(params["hidden_layers"])

    if model_name == "MC Dropout":
        cfg = MCDropoutConfig(
            layer_sizes=[input_dim, *hidden, 2],
            dropout_p=params["dropout_p"],
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            lr=params["lr"],
            mc_samples=params["mc_samples"],
            seed=params["seed"],
        )
        model = MCDropoutNet.from_config(cfg)
        model = train_mc_dropout(model, X_train, y_train, cfg)
        y_pred, y_std = predict_mc_dropout(model, X_test, cfg)
        return y_pred.reshape(-1), y_std.reshape(-1), asdict(cfg)

    if model_name == "VI-BB":
        cfg = VIBBConfig(
            layer_sizes=[input_dim, *hidden, 1],
            activation=params["activation"],
            prior_sigma_1=params["prior_sigma_1"],
            prior_sigma_2=params["prior_sigma_2"],
            prior_pi=params["prior_pi"],
            kl_weight=params["kl_weight"],
            noise_std=params["noise_std"],
            lr=params["lr"],
            epochs=params["epochs"],
            mc_samples=params["mc_samples"],
            seed=params["seed"],
        )
        model = BayesianMLP.from_config(cfg)
        model.train_model(X_train, y_train.reshape(-1, 1), cfg)
        y_pred, y_std = model.predict(X_test, cfg)
        return y_pred.reshape(-1), y_std.reshape(-1), asdict(cfg)

    if model_name == "PBP":
        cfg = PBPConfig(
            layer_sizes=[input_dim, *hidden, 1],
            n_epochs=params["n_epochs"],
            normalize=False,
            seed=params["seed"],
        )
        model = PBP_net.from_config(X_train, y_train, cfg)
        y_pred, y_var, y_noise = model.predict(X_test)
        y_std = np.sqrt(np.clip(y_var + y_noise, 1e-8, None))
        return y_pred.reshape(-1), y_std.reshape(-1), asdict(cfg)

    if model_name == "HMC":
        cfg = HMCConfig(
            layer_sizes=[input_dim, *hidden, 1],
            step_size=params["step_size"],
            num_samples=params["num_samples"],
            num_steps_per_sample=params["num_steps_per_sample"],
            tau_out=params["tau_out"],
            tau_prior=params["tau_prior"],
            burn_frac=params["burn_frac"],
            seed=params["seed"],
        )
        net = HMCNet.from_config(cfg)
        params_hmc = sample_hmc(net, X_train, y_train.reshape(-1, 1), cfg)
        y_pred, y_std = predict_hmc(net, X_test, None, params_hmc, cfg)
        return y_pred.reshape(-1), y_std.reshape(-1), asdict(cfg)

    cfg = ABCSSConfig(
        layer_sizes=[input_dim, *hidden, 1],
        activations=[params["activation"]] * (len(hidden) + 1),
        n_samples=params["n_samples"],
        sim_levels=params["sim_levels"],
        p0=params["p0"],
        initial_std=params["initial_std"],
        prior_low=params["prior_low"],
        prior_high=params["prior_high"],
        seed=params["seed"],
    )
    model = ABCSubSim(cfg)
    model.fit(X_train, y_train.reshape(-1, 1))
    y_pred, y_std = model.predict(X_test, n_best=params["n_best"])
    return y_pred.reshape(-1), y_std.reshape(-1), asdict(cfg)


# ──────────────────────────────────────────────────────────────────────────────
# Manual hyperparameter controls
# ──────────────────────────────────────────────────────────────────────────────

def model_settings_manual(model_name: str) -> dict:
    st.subheader("Architecture")
    hidden_layers = st.text_input(
        "Hidden layers", value="32,16",
        help="Comma-separated list. Example: 64,32 defines two hidden layers.",
    )
    seed = st.number_input("Seed", value=42, step=1)

    if model_name == "MC Dropout":
        st.markdown(
            """
**Hyperparameters (MC Dropout)**
- `dropout_p`: probability of dropping neurons; controls epistemic uncertainty.
- `epochs`: number of full training passes.
- `batch_size`: batch size per update.
- `lr`: learning rate.
- `mc_samples`: number of stochastic passes at inference.
"""
        )
        return {
            "hidden_layers": hidden_layers,
            "seed":          int(seed),
            "dropout_p":     st.slider("dropout_p", 0.05, 0.9, 0.2, 0.05),
            "epochs":        st.number_input("epochs", min_value=10, value=150, step=10),
            "batch_size":    st.number_input("batch_size", min_value=4, value=32, step=4),
            "lr":            st.number_input("lr", min_value=1e-5, value=1e-3, step=1e-4, format="%.5f"),
            "mc_samples":    st.number_input("mc_samples", min_value=10, value=100, step=10),
        }

    if model_name == "VI-BB":
        st.markdown(
            """
**Hyperparameters (VI-BB)**
- `activation`: non-linear activation in hidden layers.
- `prior_sigma_1`, `prior_sigma_2`, `prior_pi`: Gaussian mixture prior for weights.
- `kl_weight`: weight of the KL term (regularizes posterior vs prior).
- `noise_std`: standard deviation of the Gaussian likelihood noise.
- `lr`, `epochs`, `mc_samples`: control optimization and inference.
"""
        )
        return {
            "hidden_layers":  hidden_layers,
            "seed":           int(seed),
            "activation":     st.selectbox("activation", ["tanh", "relu"]),
            "prior_sigma_1":  st.number_input("prior_sigma_1", min_value=0.01, value=1.5, step=0.05),
            "prior_sigma_2":  st.number_input("prior_sigma_2", min_value=0.01, value=0.1, step=0.01),
            "prior_pi":       st.slider("prior_pi", 0.01, 0.99, 0.5, 0.01),
            "kl_weight":      st.number_input("kl_weight", min_value=1e-4, value=1.0, step=0.1, format="%.4f"),
            "noise_std":      st.number_input("noise_std", min_value=1e-3, value=1.0, step=0.1, format="%.3f"),
            "lr":             st.number_input("lr", min_value=1e-5, value=0.01, step=1e-3, format="%.5f"),
            "epochs":         st.number_input("epochs", min_value=50, value=600, step=50),
            "mc_samples":     st.number_input("mc_samples", min_value=10, value=100, step=10),
        }

    if model_name == "PBP":
        st.markdown(
            """
**Hyperparameters (PBP)**
- `n_epochs`: number of Bayesian update iterations.
"""
        )
        return {
            "hidden_layers": hidden_layers,
            "seed":          int(seed),
            "n_epochs":      st.number_input("n_epochs", min_value=1, value=25, step=1),
        }

    if model_name == "HMC":
        st.markdown(
            """
**Hyperparameters (HMC)**
- `step_size`: step size of the Hamiltonian integrator.
- `num_samples`: number of posterior samples.
- `num_steps_per_sample`: leapfrog steps per leapfrog step.
- `tau_out`: output noise precision (likelihood).
- `tau_prior`: prior precision on weights.
- `burn_frac`: initial fraction discarded as burn-in.
"""
        )
        return {
            "hidden_layers":        hidden_layers,
            "seed":                 int(seed),
            "step_size":            st.number_input("step_size", min_value=1e-5, value=0.0015, step=1e-4, format="%.5f"),
            "num_samples":          st.number_input("num_samples", min_value=20, value=120, step=20),
            "num_steps_per_sample": st.number_input("num_steps_per_sample", min_value=5, value=10, step=1),
            "tau_out":              st.number_input("tau_out", min_value=0.01, value=100.0, step=1.0),
            "tau_prior":            st.number_input("tau_prior", min_value=0.01, value=1.0, step=0.1),
            "burn_frac":            st.slider("burn_frac", 0.05, 0.9, 0.5, 0.05),
        }

    st.markdown(
        """
**Hyperparameters (ABC-SS)**
- `n_samples`: number of samples per level (higher = more cost).
- `sim_levels`: number of subset simulation levels.
- `p0`: elite sample fraction per level.
- `initial_std`: initial proposal std for MCMC at each level.
- `prior_low`, `prior_high`: truncation bounds for weight prior.
- `n_best`: best samples used for final prediction.
"""
    )
    return {
        "hidden_layers": hidden_layers,
        "seed":          int(seed),
        "activation":    st.selectbox("activation", ["tanh", "relu", "sigmoid"]),
        "n_samples":     st.number_input("n_samples", min_value=200, value=2500, step=100),
        "sim_levels":    st.number_input("sim_levels", min_value=1, value=3, step=1),
        "p0":            st.slider("p0", 0.05, 0.5, 0.2, 0.05),
        "initial_std":   st.number_input("initial_std", min_value=0.01, value=0.4, step=0.05, format="%.2f"),
        "prior_low":     st.number_input("prior_low", value=-1.0, step=0.1),
        "prior_high":    st.number_input("prior_high", value=1.0, step=0.1),
        "n_best":        st.number_input("n_best", min_value=10, value=200, step=10),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Optuna search-space controls
# ──────────────────────────────────────────────────────────────────────────────

def model_settings_optuna() -> dict:
    col_a, col_b = st.columns(2)
    with col_a:
        n_trials = st.number_input(
            "Number of trials", min_value=5, max_value=500, value=30, step=5,
            help="More trials → better search, longer runtime.",
        )
        alpha = st.slider(
            "Coverage α", min_value=0.01, max_value=0.20, value=0.05, step=0.01,
            help="α=0.05 targets 95 % coverage for PICP / Winkler.",
        )
    with col_b:
        _OBJ_OPTIONS = ["RMSE", "MAE", "PICP loss", "MPIW", "NLL", "Winkler"]
        _OBJ_HELP = {
            "RMSE":      "Point accuracy — lower is better",
            "MAE":       "Robust point accuracy",
            "PICP loss": "(PICP − target)² — pushes coverage to 1−α",
            "MPIW":      "Interval width — narrower means sharper bands",
            "NLL":       "Gaussian NLL — balances accuracy and calibration",
            "Winkler":   "Width + miscoverage penalty — overall calibration",
        }
        selected_objectives = st.multiselect(
            "Objectives to minimise",
            options=_OBJ_OPTIONS,
            default=["RMSE", "PICP loss", "MPIW", "NLL"],
            format_func=lambda o: f"{o}  —  {_OBJ_HELP[o]}",
            help="1 objective → TPE sampler.  2+ objectives → NSGA-II (Pareto front).",
        )

    st.info(
        "Optuna will automatically explore the architecture (number and size of "
        "hidden layers) and all model-specific hyperparameters. No manual values needed."
    )
    return {
        "n_trials":   int(n_trials),
        "alpha":      float(alpha),
        "objectives": selected_objectives,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="Demo Bayesian-NN", layout="wide")
    st.title("Bayesian Neural Networks")
    st.write(
        "Visualize the concrete dataset, clean the data, select a BNN model, "
        "tune hyperparameters, and run inference with metrics."
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    st.subheader("Dataset")
    dataset_mode = st.radio("Dataset source", ["Use default dataset", "Upload dataset"])

    if dataset_mode == "Upload dataset":
        uploaded_file = st.file_uploader(
            "Upload a dataset", type=["xls", "xlsx", "csv"],
            help="Supported formats: .xls, .xlsx, .csv",
        )
        if uploaded_file is None:
            st.info("Upload a file to continue.")
            return
        df = load_and_clean_uploaded_dataset(uploaded_file.getvalue(), uploaded_file.name)
        st.caption(f"Loaded dataset: `{uploaded_file.name}`")
    else:
        df = load_and_clean_dataset(DATASET_PATH)
        st.caption(f"Loaded dataset: `{DATASET_PATH.name}`")

    all_columns = df.columns
    target_default = all_columns[-1]

    st.subheader("Clean dataset")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows",       f"{len(df):,}")
    c2.metric("Columns",    f"{df.shape[1]:,}")
    c3.metric("Duplicates", f"{df.is_duplicated().sum():,}")
    st.dataframe(df.head(20), use_container_width=True)

    with st.expander("Select features and target"):
        target_col = st.selectbox(
            "Target", options=all_columns, index=all_columns.index(target_default)
        )
        feature_cols = [c for c in all_columns if c != target_col]
        selected_features = st.multiselect(
            "Features", options=feature_cols, default=feature_cols,
            help="Disable columns to compare model sensitivity.",
        )

    if not selected_features:
        st.warning("Select at least one feature to train the model.")
        return
        
    # ── Dataset split & preprocessing ────────────────────────────────────────
    st.subheader("Dataset split")
    test_size = st.slider("test_size", 0.1, 0.5, 0.2, 0.05)
    split_seed = st.number_input(
        "split_seed",
        min_value=0,
        value=42,
        step=1,
        help="Random seed used for train/test split in both manual and Optuna modes.",
    )
    st.markdown("---")

    st.subheader("Preprocessing")
    scale_X = st.checkbox("Scale Features (X)", value=True)
    scale_y = st.checkbox("Scale Target (y)", value=True)

    # ── Model selection ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Model Configuration")

    model_name_col, info_col_button = st.columns([11, 1], vertical_alignment="bottom")
    with model_name_col:
        model_name = st.selectbox("Model", MODEL_OPTIONS)
    with info_col_button:
        st.link_button("ℹ️ Theory", url=NOTEBOOK_URLS[model_name])

    # ── Hyperparameter mode ───────────────────────────────────────────────────
    st.markdown("#### Hyperparameter mode")
    hp_mode = st.radio(
        "hp_mode",
        options=["✋ Manual", "🔬 Optimise with Optuna"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if hp_mode == "✋ Manual":
        params = model_settings_manual(model_name)
    else:
        optuna_cfg = model_settings_optuna()
        show_metric_explanations(optuna_cfg["alpha"])



    # ── Shared data preparation (used by manual + Optuna) ────────────────────
    def _prepare_data():
        X = np.asarray(df[selected_features].to_numpy(), dtype=np.float64)
        y = np.asarray(df[target_col].to_numpy(), dtype=np.float64)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(split_seed)
        )
        _scaler_y = None
        if scale_X:
            sx = StandardScaler()
            X_train_m = sx.fit_transform(X_train)
            X_test_m  = sx.transform(X_test)
        else:
            X_train_m, X_test_m = X_train, X_test
        if scale_y:
            _scaler_y = StandardScaler()
            y_train_m = _scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        else:
            y_train_m = y_train
        return X_train_m, X_test_m, y_train_m, y_test, _scaler_y

    # ── Manual mode ───────────────────────────────────────────────────────────
    if hp_mode == "✋ Manual":


        if st.button("▶ Train and run inference", type="primary"):
            try:
                X_train_m, X_test_m, y_train_m, y_test, _scaler_y = _prepare_data()
                with st.spinner("Training model and running inference..."):
                    y_pred, y_std, cfg_dict = run_model(
                        model_name, X_train_m, X_test_m, y_train_m, params
                    )
                if scale_y and _scaler_y is not None:
                    y_pred = _scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                    y_std  = y_std * float(_scaler_y.scale_[0])

                metrics = compute_metrics(y_test, y_pred, y_std, alpha=0.05)
                show_metric_explanations(0.05)
                render_results(y_test, y_pred, y_std, cfg_dict, metrics, alpha=0.05)

            except Exception as exc:
                st.error(f"An error occurred during execution: {exc}")

    # ── Optuna mode ───────────────────────────────────────────────────────────
    else:
        if not optuna_cfg["objectives"]:
            st.warning("Select at least one objective to run the optimisation.")
            return

        if st.button("▶ Run Optuna optimisation", type="primary"):
            try:
                X_train_m, X_test_m, y_train_m, y_test, _scaler_y = _prepare_data()

                render_hpo_section(
                    model_name=model_name,
                    X_train=X_train_m,
                    X_test=X_test_m,
                    y_train=y_train_m,
                    y_test=y_test,
                    run_model_fn=run_model,
                    scaler_y=_scaler_y if scale_y else None,
                    n_trials=optuna_cfg["n_trials"],
                    alpha=optuna_cfg["alpha"],
                    objectives=optuna_cfg["objectives"],
                )
            except Exception as exc:
                st.error(f"An error occurred during execution: {exc}")


if __name__ == "__main__":
    main()