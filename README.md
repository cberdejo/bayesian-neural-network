# Bayesian Neural Network Implementations
[![Python](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)

Reference implementations and clear documentation of state-of-the-art Bayesian Neural Network (BNN) methods. Suited for safety-critical and high-stakes applications where predictive uncertainty matters more than a single point estimate.

---

## Implemented Methods

| Method | Description |
|--------|-------------|
| **ABC-SS** | Approximate Bayesian Computation by Subset Simulation |
| **HMC** | Hamiltonian Monte Carlo |
| **MC Dropout** | Monte Carlo Dropout |
| **PBP** | Probabilistic Backpropagation |
| **VI-BB** | Variational Inference for Bayesian Neural Networks (Bayes by Backprop) |

---

## Repository Structure

```
bayesian-nn/
├── src/
│   ├── notebooks/     # Explanatory notebooks per method
│   │   ├── abc_ss
│   │   ├── abcss_hmc_vi
│   │   ├── hmc
│   │   ├── mc_dropout
│   │   ├── pbp
│   │   └── vi_bbb
│   ├── packages/     # Reusable BNN algorithm implementations (see below)
│   │   ├── abc_ss
│   │   ├── hmc
│   │   ├── mc_dropout
│   │   ├── pbp
│   │   └── vi_bb
│   └── demo/         # Streamlit demo application
├── dataset/          # Default dataset and assets
└── pyproject.toml
```

### The `src/packages` Directory

`src/packages` holds the core BNN implementations as importable Python packages. Each subpackage provides a **model**, a **config** (Pydantic), and a consistent interface for training and prediction. The demo and notebooks depend on these packages.

| Package | Main exports | Role |
|---------|--------------|------|
| **abc_ss** | `ABCSubSim`, `ABCSSConfig` | Approximate Bayesian Computation via subset simulation for posterior sampling. |
| **hmc** | `HMCNet`, `HMCConfig`, `sample_hmc`, `predict_hmc` | Hamiltonian Monte Carlo sampling for Bayesian neural networks (e.g. via hamiltorch). |
| **mc_dropout** | `MCDropoutNet`, `MCDropoutConfig`, `train_mc_dropout`, `predict_mc_dropout` | Monte Carlo Dropout: uncertainty from multiple forward passes with dropout enabled at test time. |
| **pbp** | `PBP_net`, `PBP`, `PBPConfig`, `Network`, `Network_layer`, `Prior` | Probabilistic Backpropagation: analytical approximate inference over network weights. |
| **vi_bb** | `BayesianMLP`, `DenseVariational`, `VIBBConfig`, `nll_gaussian` | Variational inference (Bayes by Backprop) with reparameterized Gaussian posteriors. |

Use these packages in your own scripts by importing from `src.packages` (with the project root on `PYTHONPATH`) or by installing the project as a package.

---

## Running the Demo

The project includes a **Justfile** recipe that runs an interactive Streamlit demo on the **Concrete Compressive Strength** dataset.

### Prerequisites

- **Python** >= 3.14 (see `pyproject.toml`)
- **just** — command runner. On Ubuntu/Debian: `sudo apt install just`
- **uv** — Python environment and dependency manager: [github.com/astral-sh/uv](https://github.com/astral-sh/uv)

### Launch

From the repository root:

```bash
just demo
```

This runs:

```bash
uv sync
uv run streamlit run ./src/demo/streamlit_app.py
```

### Demo Features

- Load and preprocess tabular data (default or user-uploaded).
- Choose a BNN model: MC Dropout, VI-BB, PBP, HMC, or ABC-SS.
- Tune hyperparameters from the sidebar.
- Train and inspect:
  - Metrics: RMSE, MAE, R², NLL, confidence interval coverage.
  - Prediction plots with uncertainty.

### Default Dataset

With **"Use default dataset"** selected, the app loads the [Concrete Compressive Strength](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength) dataset from `dataset/Concrete_Data.xls`, so you can try all models without preparing data.

![Concrete strength dataset](dataset/concrete_strength.png)

You can also upload your own file (CSV, XLS, or XLSX) and run the same training and evaluation pipeline.

---

## License

MIT License. Copyright (c) 2026 Christian Berdejo Sánchez.

See [LICENSE](LICENSE) for the full text.
