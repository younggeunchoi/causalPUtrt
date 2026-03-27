# causalPUtrt

ATE (Average Treatment Effect) estimation when treatment is observed through **Positive-Unlabeled (PU) labels**.

## Problem Setup

We observe `(X, S, Y)` where:

- `X`: covariates
- `S`: PU label (`S=1` means the unit is *labeled*, i.e., known to be treated)
- `Y`: outcome
- `A`: true treatment status (**unobserved** for `S=0` units)

The key relationship: `S = 1` implies `A = 1`, but `S = 0` can mean either `A = 0` or `A = 1` (unlabeled treated).

**Goal**: Estimate ATE = E[Y(1) - Y(0)] using an estimate of π_A(X) = P(A=1|X).

## Two Settings

| Setting | Description | Estimator | Convenience wrapper |
|---------|-------------|-----------|---------------------|
| **One-sample** | All units from one population | `proposed_estimator_onesample()` | `ate_onesample()` |
| **Two-sample** | Pop 0 (S=0) and Pop 1 (S=1) sampled separately (case-control) | `proposed_estimator_twosample()` | `ate_twosample()` |

## Quick Start

The core of this package is the AIPW estimator that takes `piA_hat` — an estimate of P(A=1|X) — and returns an ATE estimate. How you obtain `piA_hat` is up to you: SAR-EM, SCAR, or any PU learning method.

```r
source("R/causalPUtrt.R")

# You provide: piA_hat, piS_hat, mu1_hat (and mu_hat for two-sample)
# The estimator does the rest.

# One-sample
result <- proposed_estimator_onesample(S, Y, piA_hat, piS_hat, split_idx, mu1_hat)

# Two-sample (case-control)
result <- proposed_estimator_twosample(S, Y, piA_hat, piS_hat, mu_hat, mu1_hat, split_idx)
```

For a quick end-to-end run, use the convenience wrappers which handle sample splitting and nuisance estimation internally:

```r
# With your own piA_hat
res <- ate_onesample(X, S, Y, piA_hat = my_piA_hat)

# Or let the wrapper run SAR-EM for you (requires Python; see Installation)
# init_sarem("/path/to/venv", "/path/to/sarpu")
# res <- ate_onesample(X, S, Y, pu_method = "sarem")
```

See `examples/example_onesample.R` and `examples/example_twosample.R` for full working examples with data generation.

## Usage

### Estimators (core interface)

The estimators take pre-computed nuisance parameters and return ATE with standard error and confidence interval. You control every step: how `piA_hat` is estimated, whether to use sample splitting, and how nuisance parameters are fitted.

```r
# One-sample
split_idx <- sample(1:n, n/2)
# ... estimate piA_hat, piS_hat, mu1_hat on Set 1 ...
result <- proposed_estimator_onesample(S, Y, piA_hat, piS_hat, split_idx, mu1_hat)
# result: named vector with est, se, ci_lower, ci_upper

# Two-sample
result <- proposed_estimator_twosample(S, Y, piA_hat, piS_hat, mu_hat, mu1_hat, split_idx)
```

Naive estimators (treating S as A) are also provided for comparison:

```r
result <- naive_estimator_onesample(S, Y, piS_hat, split_idx, mu1_hat, mu0_hat)
result <- naive_estimator_twosample(S, Y, piS_hat, split_idx, mu1_hat, mu0_hat)
```

### Convenience Wrappers

`ate_onesample()` and `ate_twosample()` bundle sample splitting, nuisance estimation, and ATE computation into a single call. Useful for quick experiments, but not the only way to use this package.

```r
# Provide piA_hat directly — wrapper handles splitting and nuisance estimation
res <- ate_onesample(X, S, Y, piA_hat = my_piA_hat)

# Or let the wrapper also run SAR-EM to estimate piA_hat
res <- ate_onesample(X, S, Y, pu_method = "sarem")
```

If you need to control sample splitting, nuisance model specification, or use piA_hat from an external source that was already fitted on a separate sample, use the core estimators directly.

### SAR-EM Setup

This package includes SAR-EM as a built-in PU method for estimating `piA_hat`, but `piA_hat` can come from any source — SCAR, other PU methods, or domain knowledge.

To use SAR-EM, initialize the Python environment after installing it (see next section): 

```r
init_sarem("/path/to/venv", "/path/to/SAR-PU/sarpu")

# Use via wrapper
res <- ate_onesample(X, S, Y, pu_method = "sarem")

# Or standalone
sarem_fit <- fit_sarem(X, S, max_its = 500)
preds     <- predict_sarem(sarem_fit, X)
piA_hat   <- preds$prob_pred
```

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/causalPUtrt.git
cd causalPUtrt
```

```r
source("R/causalPUtrt.R")
```

The core estimators have no external dependencies. SAR-EM requires additional setup:

### SAR-EM (Python dependency)

1. **R package**:

    ```r
    install.packages("reticulate")
    ```

2. **Python environment with sarpu installed** (requires **Python >= 3.10** and **virtualenv**):

    ```bash
    virtualenv -p python3 /path/to/your/venv
    source /path/to/your/venv/bin/activate

    git clone https://github.com/ML-KULeuven/SAR-PU.git
    cd SAR-PU
    pip install -r requirements.txt
    pip install -e sarpu/
    ```

    > **Note (scikit-learn >= 1.7):** The original SAR-PU code uses deprecated parameters (`multi_class`, `penalty`, `n_jobs`). After cloning, replace the `LogisticRegressionPU` class in `SAR-PU/sarpu/sarpu/PUmodels.py` with:
    >
    > ```python
    > class LogisticRegressionPU(LogisticRegression, BasePU):
    >     def __init__(self, dual=False, tol=1e-4, C=1.0,
    >                  fit_intercept=True, intercept_scaling=1, class_weight=None,
    >                  random_state=None, solver='liblinear', max_iter=100,
    >                  verbose=0, warm_start=False):
    >         LogisticRegression.__init__(self, dual=dual, tol=tol, C=C,
    >                          fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
    >                         class_weight=class_weight, random_state=random_state,
    >                         solver=solver, max_iter=max_iter,
    >                         verbose=verbose, warm_start=warm_start)
    > ```

3. **In R, initialize before use**:

    ```r
    init_sarem(
      python_env = "/path/to/your/venv",
      sarpu_path = "/path/to/SAR-PU/sarpu"
    )
    ```

## Available Functions

| Function | Role | Description |
|----------|------|-------------|
| `proposed_estimator_onesample()` | **Estimator** | Proposed DR estimator (one-sample), takes `piA_hat` directly |
| `proposed_estimator_twosample()` | **Estimator** | Proposed DR estimator (two-sample / case-control) |
| `naive_estimator_onesample()` | **Estimator** | Naive DR treating S as A |
| `naive_estimator_twosample()` | **Estimator** | Naive two-sample DR |
| `ate_onesample()` | Wrapper | Convenience: split + nuisance + ATE (`piA_hat` or `pu_method = "sarem"`) |
| `ate_twosample()` | Wrapper | Same for two-sample |

## File Structure

```
causalPUtrt/
├── R/
│   ├── causalPUtrt.R       # Main entry point (source this)
│   ├── ate_estimators.R    # ATE estimator functions (core)
│   ├── ate_wrappers.R      # Convenience wrapper functions
│   ├── pu_sarem.R          # SAR-EM wrapper (Python)
│   └── pu_utils.R          # Utility functions
├── examples/
│   ├── example_onesample.R     # One-sample example
│   └── example_twosample.R     # Two-sample example
└── README.md
```

## References

- Bekker, J., & Davis, J. (2020). Learning from positive and unlabeled data: A survey. *Machine Learning*, 109, 719–760.
- Bekker, J., & Davis, J. (2019). Beyond the selected completely at random assumption for learning from positive and unlabeled data. *ECML-PKDD 2019*. arXiv:1809.03207. **(SAR-EM)**
