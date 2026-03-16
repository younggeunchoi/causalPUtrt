# causalPUtrt

ATE (Average Treatment Effect) estimation when treatment is observed through **Positive-Unlabeled (PU) labels**.

## Problem Setup

We observe `(X, S, Y)` where:

- `X`: covariates
- `S`: PU label (`S=1` means the unit is *labeled*, i.e., known to be treated)
- `Y`: outcome
- `A`: true treatment status (**unobserved** for `S=0` units)

The key relationship: `S = 1` implies `A = 1`, but `S = 0` can mean either `A = 0` or `A = 1` (unlabeled treated).

**Goal**: Estimate ATE = E[Y(1) - Y(0)] using PU learning to recover P(A=1|X).

## Two Settings

| Setting | Description | Wrapper | Low-level estimator |
|---------|-------------|---------|---------------------|
| **One-sample** | All units from one population | `ate_onesample()` | `proposed_estimator_onesample()` |
| **Two-sample** | Pop 0 (S=0) and Pop 1 (S=1) sampled separately (case-control) | `ate_twosample()` | `proposed_estimator_twosample()` |

## Quick Start

### One-Sample

```r
library(MASS)
source("R/causalPUtrt.R")

# --- DGP (beta0=0, gamma=0.8) ---
n <- 1500
mu_X <- c(1, -1)
Sigma_X <- matrix(c(1, 0.2, 0.2, 1), 2, 2)
X <- mvrnorm(n, mu_X, Sigma_X)
colnames(X) <- c("X1", "X2")

piA_true <- plogis(X[,1] + X[,2])            # beta0=0, beta=1
A <- rbinom(n, 1, piA_true)
e_true <- plogis(0.8 * (X[,1] + X[,2]))      # gamma=0.8
S <- A * rbinom(n, 1, e_true)
Y0 <- X[,1] - X[,2] + rnorm(n)
Y1 <- Y0 + 0.5 + 0.25*X[,1] - 0.25*X[,2]
Y  <- ifelse(A == 0, Y0, Y1)
# True ATE = 1.0

# TM
res_tm <- ate_onesample(X, S, Y, pu_method = "tm")
cat("TM:     est =", round(res_tm$est, 3), "\n")

# SAR-EM (requires Python sarpu; see Installation)
# init_sarem("/path/to/venv", "/path/to/sarpu")
# res_sarem <- ate_onesample(X, S, Y, pu_method = "sarem")
# cat("SAR-EM: est =", round(res_sarem$est, 3), "\n")
```

### Two-Sample

```r
library(MASS)
source("R/causalPUtrt.R")

# --- DGP (m=0.2, n1=500, case-control) ---
n0 <- 1000; n1 <- 500; pA_pop0 <- 0.3
Sigma <- matrix(c(1, 0.2, 0.2, 1), 2, 2)

A0 <- rbinom(n0, 1, pA_pop0)
X0 <- matrix(NA, n0, 2)
X0[A0==1,] <- mvrnorm(sum(A0), c(0, 0), Sigma)
X0[A0==0,] <- mvrnorm(n0 - sum(A0), c(0.5, -0.5), Sigma)

X1 <- mvrnorm(n1, c(0.2, -0.2), Sigma)
X  <- rbind(X0, X1); colnames(X) <- c("X1", "X2")
A  <- c(A0, rep(1, n1))
S  <- c(rep(0, n0), rep(1, n1))

Y0_pot <- X[,1] - X[,2] + rnorm(n0 + n1)
Y1_pot <- Y0_pot + 0.5 + 0.25*X[,1] - 0.25*X[,2]
Y <- ifelse(A == 0, Y0_pot, Y1_pot)

# TM
res_tm <- ate_twosample(X, S, Y, pu_method = "tm")
cat("TM:     est =", round(res_tm$est, 3), "\n")

# SAR-EM
# init_sarem("/path/to/venv", "/path/to/sarpu")
# res_sarem <- ate_twosample(X, S, Y, pu_method = "sarem")
# cat("SAR-EM: est =", round(res_sarem$est, 3), "\n")
```

## PU Learning Methods

Three methods are available for estimating π_A(X) = P(A=1|X):

| Method | Language | External Dependency | Function |
|--------|----------|-------------------|----------|
| **TM** | Pure R | None | `pu_learn_tm()` |
| **SAR-EM** | Python via reticulate | `sarpu` Python package | `fit_sarem()` |
| **DETM** | R | `PUEM` R package | `fit_detm()` |

## Installation

### Base (TM only)

No installation needed beyond base R + MASS:

```r
install.packages("MASS")  # for mvrnorm in examples
```

Clone and use:

```bash
git clone https://github.com/YOUR_USERNAME/causalPUtrt.git
cd causalPUtrt
```

```r
source("R/causalPUtrt.R")
```

### SAR-EM (Python dependency)

SAR-EM runs Python's `sarpu` package through `reticulate`. You need:

1. **R package**:

    ```r
    install.packages("reticulate")
    ```

2. **Python virtualenv with sarpu installed**:

    ```bash
    # Create a Python virtualenv
    virtualenv -p python3 /path/to/your/venv
    source /path/to/your/venv/bin/activate

    # Clone and install sarpu
    git clone https://github.com/ML-KULeuven/SAR-PU.git
    cd SAR-PU
    pip install -r requirements.txt
    pip install -e sarpu/
    ```

3. **In R, initialize before use**:

    ```r
    init_sarem(
      virtualenv_path = "/path/to/your/venv",
      sarpu_path      = "/path/to/SAR-PU/sarpu"
    )
    ```

### DETM (PUEM R package)

DETM uses the `PUEM` R package, which is not on CRAN. Install from GitHub:

```r
# Option 1: Install from the PUEM source directory
R CMD INSTALL /path/to/PUEM/PUEM

# Option 2: If available via devtools
# devtools::install_github("AUTHOR/PUEM")
```

The PUEM package depends on `glmnet`:

```r
install.packages("glmnet")
```

## Advanced Usage

### Low-Level Estimators

The wrapper functions (`ate_onesample`, `ate_twosample`) handle the full pipeline. For custom pipelines, use the low-level estimators directly:

```r
# One-sample: manual pipeline
split_idx <- sample(1:n, n/2)
tm_fit  <- pu_learn_tm(X[split_idx,], S[split_idx], epochs = 500)
piA_hat <- predict_proba_pu(tm_fit$model_clf, X)
# ... estimate piS_hat, mu1_hat on Set 1 ...
result <- proposed_estimator_onesample(S, Y, piA_hat, piS_hat, split_idx, mu1_hat)

# Two-sample: manual pipeline
# ... estimate piA_hat, piS_hat, mu_hat, mu1_hat on Set 1 ...
result <- proposed_estimator_twosample(S, Y, piA_hat, piS_hat, mu_hat, mu1_hat, split_idx)
```

### Using SAR-EM or DETM Directly

```r
# SAR-EM
init_sarem("/path/to/venv", "/path/to/SAR-PU/sarpu")
sarem_fit <- fit_sarem(X_train, S_train, max_its = 500)
preds     <- predict_sarem(sarem_fit, X)
piA_hat   <- preds$prob_pred

# DETM
library(PUEM)
detm_fit <- fit_detm(X_train, S_train)
piA_hat  <- predict_detm(detm_fit, X)
```

## Available Functions

| Function | Description |
|----------|-------------|
| `ate_onesample()` | End-to-end one-sample ATE (wrapper) |
| `ate_twosample()` | End-to-end two-sample ATE (wrapper) |
| `proposed_estimator_onesample()` | Proposed DR estimator (one-sample) |
| `proposed_estimator_twosample()` | Proposed DR estimator (two-sample) |
| `naive_estimator_onesample()` | Naive DR treating S as treatment |
| `naive_estimator_twosample()` | Naive two-sample DR estimator |

## File Structure

```
causalPUtrt/
├── R/
│   ├── causalPUtrt.R       # Main entry point (source this)
│   ├── ate_wrappers.R      # End-to-end wrapper functions
│   ├── ate_estimators.R    # ATE estimator functions
│   ├── pu_tm.R             # TM method (pure R)
│   ├── pu_sarem.R          # SAR-EM wrapper (Python)
│   ├── pu_detm.R           # DETM wrapper (PUEM)
│   ├── pu_models.R         # Logistic regression for PU
│   ├── pu_utils.R          # Utility functions
│   └── irwls.R             # IRWLS solver with L2 regularization
├── examples/
│   ├── example_onesample.R     # One-sample with TM
│   ├── example_twosample.R     # Two-sample with TM
│   └── example_all_methods.R   # Compare SAR-EM, TM, DETM
└── README.md
```

## References

- Bekker, J., & Davis, J. (2020). Learning from positive and unlabeled data: A survey. *Machine Learning*, 109, 719–760.
