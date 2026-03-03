# Bayesian Neural Network Implementations

This repository provides clear explanations and reference implementations of state-of-the-art Bayesian Neural Network (BNN) models. These methods are particularly useful in safety-critical and high-stakes applications, where it is important to quantify predictive uncertainty rather than returning single point estimates.

The following BNN approaches are implemented and explained:
- Approximate Bayesian Computation by Subset Simulation (ABC-SS)
- Hamiltonian Monte Carlo (HMC)
- Monte Carlo Dropout
- Probabilistic Backpropagation (PBP)
- Variational Inference for Bayesian Neural Networks (VI)

## Repo Architecture
```bash 
├── src
│   ├── notebooks # explanations of model
│   │   ├── abc_ss
│   │   ├── abcss_hmc_vi
│   │   ├── hmc
│   │   ├── mc_dropout
│   │   ├── pbp
│   │   └── vi_bbb
│   ├── app # user case with real data
```