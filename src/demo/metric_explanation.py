import streamlit as st

def show_metric_explanations(alpha: float) -> None:

    with st.expander("What do these metrics mean?"):
        ci_pct = int(round((1 - alpha) * 100))
        st.markdown(
            f"""
- **RMSE**: Root-mean-squared error of the predictive mean; lower is better.
- **PICP**: Fraction of true targets inside the {ci_pct} % predictive interval; closer to {ci_pct/100:.0%} is better.
- **MPIW**: Average width of the predictive interval; narrower means sharper predictions.
- **NLL**: Gaussian negative log-likelihood, combining accuracy and uncertainty calibration.
- **Winkler**: Proper scoring rule that penalises intervals that are too wide or miss the true value.
"""
        )