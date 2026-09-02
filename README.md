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
| **Two-sample** | Pop 0 (S=0) and Pop 1 (S=1) sampled separately (case-control); target τ^UL = E(Y¹−Y⁰ \| S=0) | `proposed_estimator_twosample()` | `ate_twosample()` |

## Quick Start

```r
source("R/causalPUtrt.R")

# SCAR (pure R; requires the known labeling rate c = P(S=1|A=1))
res <- ate_onesample(X, S, Y, pu_method = "scar", c_known = c)

# SAR-EM (requires Python; see Installation)
# init_sarem("/path/to/venv", "/path/to/SAR-PU/sarpu")
# res <- ate_onesample(X, S, Y, pu_method = "sarem")

# Or supply your own piA_hat (theta-correction omitted; see the warning below)
res <- ate_onesample(X, S, Y, piA_hat = my_piA_hat)

res$est; res$se; c(res$ci_lower, res$ci_upper)
```

See `examples/example_onesample.R` and `examples/example_twosample.R` for full working examples with data generation.


## Usage

### Convenience Wrappers

`ate_onesample()` and `ate_twosample()` bundle sample splitting, π_A estimation, nuisance estimation, and the corrected SE into a single call.

```r
# SCAR: c = P(S=1|A=1) must be known (it is not identifiable from (X,S,Y) alone)
res <- ate_onesample(X, S, Y, pu_method = "scar", c_known = 0.6)

# SAR-EM (after init_sarem)
res <- ate_onesample(X, S, Y, pu_method = "sarem")

# Custom piA_hat (theta-correction omitted; see warning above)
res <- ate_onesample(X, S, Y, piA_hat = my_piA_hat)
```

Built-in π_A methods:

| Method | Backend | Extra requirement | Fitter / ξ_θ |
|--------|---------|-------------------|--------------|
| **SCAR** | Pure R | known `c = P(S=1\|A=1)` | `fit_scar_mle()` / `xi_scar_eval()` |
| **SAR-EM** | Python via reticulate | sarpu | `fit_sarem()` / `xi_sar_eval()` |

### Core Estimators (low-level)

The estimators take pre-computed nuisance values plus (optionally) the design matrix `Z_ev` and influence-function values `Xi_ev` at the eval observations:

```r
split_idx <- sample(1:n, n/2)
ev   <- setdiff(1:n, split_idx)
Z_tr <- cbind(1, X[split_idx, ]);  Z_ev <- cbind(1, X[ev, ])

# Example: SAR-EM piA with its influence function (after init_sarem)
sarem_fit <- fit_sarem(X[split_idx, ], S[split_idx], max_its = 500, C = 1.0)
piA_hat   <- predict_sarem(sarem_fit, X)$prob_pred
ridge     <- 1 / (length(split_idx) * 1.0)   # 1/(n_tr * C)
Xi_ev     <- xi_sar_eval(sarem_fit$theta, sarem_fit$phi,
                         Z_tr, Z_tr, S[split_idx],    # W_tr = Z_tr (both models
                         Z_ev, Z_ev, S[ev], ridge)    #  use all covariates)

# ... estimate piS_hat, mu1_hat (and mu_hat for two-sample) on Set 1 ...

result <- proposed_estimator_onesample(S, Y, piA_hat, piS_hat, split_idx, mu1_hat,
                                       Z_ev = Z_ev, Xi_ev = Xi_ev)
# named vector: est, se, ci_lower, ci_upper

result <- proposed_estimator_twosample(S, Y, piA_hat, piS_hat, mu_hat, mu1_hat,
                                       split_idx, Z_ev = Z_ev, Xi_ev = Xi_ev)
```

Omitting `Xi_ev` drops the θ-correction (valid only if π_A is known or estimated at o_p(n^{-1/2}) rate). Naive estimators (treating S as A) are also provided for comparison:

```r
result <- naive_estimator_onesample(S, Y, piS_hat, split_idx, mu1_hat, mu0_hat)
result <- naive_estimator_twosample(S, Y, piS_hat, split_idx, mu1_hat, mu0_hat)
```

### SAR-EM Setup

To use SAR-EM, initialize the Python environment after installing it (see next section):

```r
init_sarem("/path/to/venv", "/path/to/SAR-PU/sarpu")

# Use via wrapper
res <- ate_onesample(X, S, Y, pu_method = "sarem")

# Or standalone (coefficients only cross the R/Python boundary)
sarem_fit <- fit_sarem(X_train, S_train, max_its = 500)
piA_hat   <- predict_sarem(sarem_fit, X)$prob_pred
```

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/causalPUtrt.git
cd causalPUtrt
```

```r
source("R/causalPUtrt.R")
```

The core estimators and SCAR have no external dependencies. SAR-EM requires additional setup:

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
| `proposed_estimator_onesample()` | **Estimator** | Proposed AIPW (one-sample), corrected SE via `Z_ev`/`Xi_ev` |
| `proposed_estimator_twosample()` | **Estimator** | Proposed AIPW (two-sample / case-control), corrected SE |
| `naive_estimator_onesample()` | **Estimator** | Naive DR treating S as A |
| `naive_estimator_twosample()` | **Estimator** | Naive two-sample DR |
| `fit_scar_mle()` / `xi_scar_eval()` | π_A method | SCAR scaled-logistic MLE (known c) and its influence function |
| `fit_sarem()` / `predict_sarem()` / `xi_sar_eval()` | π_A method | SAR-EM (norefit) and its penalized-joint influence function |
| `ate_onesample()` | Wrapper | split + π_A + nuisance + corrected-SE ATE |
| `ate_twosample()` | Wrapper | Same for two-sample |

## File Structure

```
causalPUtrt/
├── R/
│   ├── causalPUtrt.R       # Main entry point (source this)
│   ├── ate_estimators.R    # AIPW estimators with corrected (Thm 3/S3) SE
│   ├── ate_wrappers.R      # Convenience wrapper functions
│   ├── pu_scar.R           # SCAR scaled-logistic MLE + xi_theta (pure R)
│   ├── pu_sarem.R          # SAR-EM norefit wrapper + xi_theta (Python)
│   └── pu_utils.R          # Utility functions
├── examples/
│   ├── example_onesample.R     # One-sample example (SCAR runs out of the box)
│   └── example_twosample.R     # Two-sample example
└── README.md
```

## Theoretical Note about Standard Errors

The reported SE is the **corrected variance** of the paper's Theorem 3 (one-sample) and Theorem S3 (two-sample). When π_A is estimated by a parametric model with coefficient estimator θ̂ satisfying a √n asymptotic linear expansion with influence function ξ_θ, the influence function of the AIPW estimator is

```
psi_par_i = psi_i + B' xi_theta_i        (one-sample; Theorem 3)
psi_par_i = psi^UL_i + B_UL' xi_theta_i  (two-sample; Theorem S3)
```

and the SE is the empirical standard deviation of `psi_par` over the estimation (eval) half, divided by √n_ev. The plug-in variance of `psi` alone (without the θ-correction) ignores the uncertainty from estimating π_A and can severely under-cover.

> **⚠ ξ_θ is method-dependent.** This package provides ξ_θ for **exactly two** π_A estimation procedures:
>
> 1. **SCAR** — the scaled-logistic MLE `piS(X;θ) = c·expit(θ'Z)` with **known** `c = P(S=1|A=1)`, which induces `piA(X;θ) = expit(θ'Z)`. Its MLE influence function is implemented in `xi_scar_eval()`.
> 2. **SAR** — SAR-EM run without the classifier refit (`refit_classifier=False`), so that (θ̂, φ̂) solves the L2-penalized joint logistic estimating equation. The θ-block of the penalized joint influence function (ridge `1/(n_tr·C)` matching sklearn's penalty) is implemented in `xi_sar_eval()`.
>
> If you estimate π_A by **any other procedure**, these ξ_θ formulas above do **not** apply — you must derive the influence function of your own estimator and pass it via the `Xi_ev` argument of the low-level estimators. When you supply `piA_hat` directly to the wrappers, the θ-correction is omitted; the reported SE is then valid only if π_A is known or estimated at o_p(n^{-1/2}) rate (condition (R1) of Theorem 2).

## References

- Bekker, J., & Davis, J. (2020). Learning from positive and unlabeled data: A survey. *Machine Learning*, 109, 719–760.
- Bekker, J., & Davis, J. (2019). Beyond the selected completely at random assumption for learning from positive and unlabeled data. *ECML-PKDD 2019*. arXiv:1809.03207. **(SAR-EM)**
