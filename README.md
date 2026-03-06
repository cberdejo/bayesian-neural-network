# Bayesian Neural Network Implementations

This repository provides clear explanations and reference implementations of state-of-the-art Bayesian Neural Network (BNN) models. These approaches are particularly useful in safety-critical and high-stakes applications, where quantifying predictive uncertainty is essential rather than returning single point estimates.

The following BNN approaches are implemented and explained:
- Approximate Bayesian Computation by Subset Simulation (ABC-SS)
- Hamiltonian Monte Carlo (HMC)
- Monte Carlo Dropout
- Probabilistic Backpropagation (PBP)
- Variational Inference for Bayesian Neural Networks (VI)

## Repository Structure
```bash 
├── src
│   ├── notebooks        # Model explanations
│   │   ├── abc_ss
│   │   ├── abcss_hmc_vi
│   │   ├── hmc
│   │   ├── mc_dropout
│   │   ├── pbp
│   │   └── vi_bbb
│   ├── packages         # Ready-to-use implemented algorithms
│   │   ├── abc_ss
│   │   ├── abcss_hmc_vi
│   │   ├── hmc
│   │   ├── mc_dropout
│   │   ├── pbp
│   │   └── vi_bbb
│   ├── demo             # Real data demonstration
```

## How to Run the Demo Using `just`

This project includes a `Justfile` with a recipe to launch an interactive Streamlit demo using the **Concrete Compressive Strength** dataset.

### Prerequisites

- **Python**: Version \(\geq 3.14\) (as specified in `pyproject.toml`)
- **just**: Command runner. On Ubuntu/Debian, install with:

```bash
sudo apt install just
```

- **uv**: Python dependency and environment manager. See instructions at `https://github.com/astral-sh/uv`.

### Launching the Demo

From the root of the repository:

```bash
just demo
```

This recipe internally runs:

```bash
uv sync
uv run streamlit run ./src/demo/streamlit_app.py
```

This ensures dependencies in `pyproject.toml` are synchronized and the Streamlit application is launched.

### About the Demo and Default Dataset

The Streamlit demo allows you to:

- Load and clean a tabular dataset
- Select a BNN model (MC Dropout, VI-BB, PBP, HMC, or ABC-SS)
- Adjust hyperparameters from the sidebar
- Train the model and visualize:
  - Metrics (RMSE, MAE, \(R^2\), NLL, confidence interval coverage)
  - Prediction plots with uncertainty




By default, if you select **"Use default dataset"** in the sidebar, the app loads the **Concrete Compressive Strength** ([UCI(https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength)]) dataset, which is stored at `dataset/Concrete_Data.xls`. This provides an out-of-the-box example for experimenting with different Bayesian architectures, with no need to prepare your own data.

[dataset-concrete](dataset/concrete_strength.png)

Alternatively, you can upload your own dataset (\*.csv, \*.xls, \*.xlsx) from the same interface and run the full training and inference workflow with your data.