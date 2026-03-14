"""
results.py – Shared inference rendering for Bayesian-NN Streamlit app.

Used by both app.py (manual mode) and hpo.py (best-trial mode).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import streamlit as st


def render_results(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    cfg_dict: dict,
    metrics: dict[str, float],
    alpha: float = 0.05,
    title: str | None = None,
) -> None:
    """
    Renders the full inference result block:
      - metric tiles (RMSE, PICP, MPIW, NLL, Winkler)
      - metric explainer expander
      - inference plot (prediction vs true + uncertainty band)
      - sortable result table
      - configuration expander

    Parameters
    ----------
    y_test   : true targets in original scale
    y_pred   : predictive mean in original scale
    y_std    : predictive std in original scale
    cfg_dict : model configuration dict (shown in expander)
    metrics  : pre-computed metrics dict from compute_metrics()
    alpha    : significance level used for the interval (only for the plot label)
    title    : optional section title rendered above the metrics
    """
    if title:
        st.subheader(title)

    # ── Metric tiles ──────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("RMSE",    f"{metrics['RMSE']:.4f}")
    m2.metric("PICP",    f"{metrics['PICP']:.4f}")
    m3.metric("MPIW",    f"{metrics['MPIW']:.4f}")
    m4.metric("NLL",     f"{metrics['NLL']:.4f}")
    m5.metric("Winkler", f"{metrics['Winkler']:.4f}")



    # ── Inference plot ────────────────────────────────────────────────────────
    st.pyplot(plot_inference(y_test, y_pred, y_std, alpha=alpha), use_container_width=True)

    # ── Result table ──────────────────────────────────────────────────────────
    st.subheader("Inference results (test samples)")
    out_df = pl.DataFrame({
        "y_true":      y_test,
        "y_pred_mean": y_pred,
        "y_pred_std":  y_std,
        "abs_error":   np.abs(y_test - y_pred),
    })
    st.dataframe(out_df.sort("abs_error", descending=True).head(30), use_container_width=True)

    # ── Config ────────────────────────────────────────────────────────────────
    with st.expander("Configuration used"):
        st.json(cfg_dict)


def plot_inference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    alpha: float = 0.05,
) -> plt.Figure:
    """
    Returns a two-panel matplotlib figure:
      left  – scatter of predicted vs true values with identity line
      right – sorted sequence with predictive mean and CI band
    """
    from scipy.stats import norm
    z = float(norm.ppf(1.0 - alpha / 2.0))
    ci_pct = int(round((1 - alpha) * 100))

    order = np.argsort(y_true)
    yt, yp, ys = y_true[order], y_pred[order], y_std[order]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: scatter
    ax0 = axes[0]
    ax0.scatter(yt, yp, alpha=0.8, edgecolor="k", linewidth=0.3)
    lo = float(min(yt.min(), yp.min()))
    hi = float(max(yt.max(), yp.max()))
    ax0.plot([lo, hi], [lo, hi], linestyle="--", color="gray")
    ax0.set_title("Prediction vs true value")
    ax0.set_xlabel("True y")
    ax0.set_ylabel("Predicted y")

    # Right: sequence + CI band
    ax1 = axes[1]
    x_axis = np.arange(len(yt))
    ax1.plot(x_axis, yt, label="True", linewidth=2)
    ax1.plot(x_axis, yp, label="Predictive mean", linewidth=2)
    ax1.fill_between(
        x_axis, yp - z * ys, yp + z * ys,
        alpha=0.25, label=f"{ci_pct} % CI",
    )
    ax1.set_title("Inference with uncertainty")
    ax1.set_xlabel("Samples (sorted by true value)")
    ax1.set_ylabel("Target")
    ax1.legend(loc="best")

    fig.tight_layout()
    return fig