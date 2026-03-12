# causalPUtrt

Doubly robust ATE (Average Treatment Effect) estimation when treatment is observed through **Positive-Unlabeled (PU) labels**.

## Problem Setup

We observe `(X, S, Y)` where:

- `X`: covariates
- `S`: PU label (`S=1` means the unit is *labeled*, i.e., known to be treated)
- `Y`: outcome
- `A`: true treatment status (**unobserved** for `S=0` units)

The key relationship: `S = 1` implies `A = 1`, but `S = 0` can mean either `A = 0` or `A = 1` (unlabeled treated).

**Goal**: Estimate ATE = E[Y(1) - Y(0)] using PU learning to recover P(A=1|X).

## Two Settings

| Setting | Description | Estimator |
|---------|-------------|-----------|
| **One-sample** | All units from one population | `proposed_estimator_onesample()` |
| **Two-sample** | Pop 0 (S=0) and Pop 1 (S=1) sampled separately (case-control) | `proposed_estimator_twosample()` |

## Quick Start (TM only, no external dependencies)

TM (Two Models) is implemented in pure R and requires no Python or external packages.

```r
# From the repo root directory:
source("R/causalPUtrt.R")

# Fit TM to estimate P(A=1|X)
tm_fit <- pu_learn_tm(X_train, S_train, epochs = 500)
piA_hat <- predict_proba_pu(tm_fit$model_clf, X)

# One-sample ATE (with sample splitting)
result <- proposed_estimator_onesample(X, S, Y, piA_hat, piS_hat,
                                       split_idx, mu1_hat)
# result contains: est, var, se, ci_lower, ci_upper

# Two-sample ATE
result <- proposed_estimator_twosample(data.frame(S=S, Y=Y),
                                       piA_hat, piS_hat, mu_hat, mu1_hat)
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

    You must know:
    - `virtualenv_path`: The path to the Python virtualenv where sarpu is installed.
    - `sarpu_path`: The path to the `sarpu/` directory inside the cloned SAR-PU repo (the directory that contains the `sarpu/` Python package).

    These paths are **machine-specific** — set them according to your environment.

### DETM (PUEM R package)

DETM uses the `PUEM` R package, which is not on CRAN. Install from GitHub:

```r
# Option 1: Install from the PUEM source directory
# Download or clone the PUEM package, then:
R CMD INSTALL /path/to/PUEM/PUEM

# Option 2: If available via devtools
# devtools::install_github("AUTHOR/PUEM")
```

The PUEM package depends on `glmnet`:

```r
install.packages("glmnet")
```

## Usage

### One-Sample Setting

```r
source("R/causalPUtrt.R")

# Your data: X (n x p matrix), S (0/1 vector), Y (outcome vector)

# Step 1: Sample split
split_idx <- sample(1:n, n/2)
X_train <- X[split_idx, ]
S_train <- S[split_idx]
Y_train <- Y[split_idx]

# Step 2: Estimate piA via PU learning on training set
tm_fit  <- pu_learn_tm(X_train, S_train, epochs = 500)
piA_hat <- predict_proba_pu(tm_fit$model_clf, X)  # predict on ALL data

# Step 3: Estimate nuisance parameters on training set
# piS = P(S=1|X)
fit_piS <- glm(S ~ ., data = data.frame(S = S_train, X_train), family = binomial())
piS_hat <- predict(fit_piS, newdata = data.frame(X), type = "response")

# mu1 = E[Y|S=1, X]
labeled <- which(S_train == 1)
fit_mu1 <- lm(Y ~ ., data = data.frame(Y = Y_train[labeled], X_train[labeled, ]))
mu1_hat <- predict(fit_mu1, newdata = data.frame(X))

# Step 4: Estimate ATE (uses Set 2 = all except split_idx)
result <- proposed_estimator_onesample(X, S, Y, piA_hat, piS_hat,
                                       split_idx, mu1_hat)
cat("ATE:", result["est"], "±", 1.96 * result["se"], "\n")
```

### Two-Sample Setting

```r
source("R/causalPUtrt.R")

# Your data: two populations already merged
# S=0 units from Pop 0, S=1 units from Pop 1

# Estimate piA, piS, mu, mu1 as above, then:
result <- proposed_estimator_twosample(
  data    = data.frame(S = S, Y = Y),
  piA_hat = piA_hat,
  piS_hat = piS_hat,
  mu_hat  = mu_hat,   # E[Y|X]
  mu1_hat = mu1_hat   # E[Y|S=1, X]
)
```

### Using SAR-EM Instead of TM

```r
# Initialize Python (once per session)
init_sarem("/path/to/venv", "/path/to/SAR-PU/sarpu")

# Fit
sarem_fit <- fit_sarem(X_train, S_train, max_its = 500)

# Predict on full data
preds   <- predict_sarem(sarem_fit, X)
piA_hat <- preds$prob_pred   # P(A=1|X)
e_hat   <- preds$prop_pred   # P(S=1|A=1,X)  (bonus)
```

### Using DETM Instead of TM

```r
library(PUEM)

detm_fit <- fit_detm(X_train, S_train)
piA_hat  <- predict_detm(detm_fit, X)
e_hat    <- predict_e_detm(detm_fit, X)  # P(S=1|A=1,X)
```

## Available Estimators

| Function | Description |
|----------|-------------|
| `proposed_estimator_onesample()` | Proposed DR estimator (one-sample) |
| `proposed_estimator_twosample()` | Proposed DR estimator (two-sample) |
| `naive_estimator_onesample()` | Naive DR treating S as treatment |
| `naive_estimator_twosample()` | Naive difference-in-means |
| `kato_estimator_onesample()` | Kato (NeurIPS 2025) estimator |

## File Structure

```
causalPUtrt/
├── R/
│   ├── causalPUtrt.R       # Main entry point (source this)
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

- Kato, M. (2025). Efficient Average Treatment Effect Estimation with Positive-Unlabeled Data. *NeurIPS 2025*.
- Bekker, J., & Davis, J. (2020). Learning from positive and unlabeled data: A survey. *Machine Learning*, 109, 719–760.
