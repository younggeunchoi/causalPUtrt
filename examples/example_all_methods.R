# example_all_methods.R
# Compare all three PU methods (SAR-EM, TM, DETM) for one-sample ATE
#
# NOTE: This example requires all three backends:
#   - TM:     pure R (always available)
#   - SAR-EM: Python sarpu package via reticulate
#   - DETM:   PUEM R package
# See README.md for installation instructions.
#
# DGP: beta0=0, gamma=0.8 (same as example_onesample.R)

library(MASS)
source("R/causalPUtrt.R")

# --- CONFIGURE PATHS (edit these for your environment) ---
SAREM_VENV_PATH  <- "/path/to/your/virtualenv"       # Python virtualenv
SAREM_SARPU_PATH <- "/path/to/SAR_PU_python/sarpu"   # sarpu source directory

# --- Parameters ---
n      <- 1500
beta0  <- 0
beta   <- 1.0
gamma  <- 0.8
tau    <- 1.0

# --- Generate data ---
set.seed(42)
mu_X    <- c(1, -1)
Sigma_X <- matrix(c(1, 0.2, 0.2, 1), 2, 2)
X       <- mvrnorm(n, mu_X, Sigma_X)
colnames(X) <- c("X1", "X2")

piA_true <- plogis(beta0 + beta * X[,1] + beta * X[,2])
A        <- rbinom(n, 1, piA_true)
e_true   <- plogis(gamma * X[,1] + gamma * X[,2])
S        <- A * rbinom(n, 1, e_true)
Y0       <- X[,1] - X[,2] + rnorm(n)
Y1       <- Y0 + tau/2 + (tau/4)*X[,1] - (tau/4)*X[,2]
Y        <- ifelse(A == 0, Y0, Y1)
ATE_true <- tau/2 + mu_X[1]*(tau/4) + mu_X[2]*(-tau/4)

cat("=== Comparing PU Methods for ATE (True ATE =", ATE_true, ") ===\n\n")

# --- Method 1: TM (always available) ---
cat("[TM] Fitting via wrapper...\n")
res_tm <- ate_onesample(X, S, Y, pu_method = "tm", epochs = 500)
cat("[TM] est =", round(res_tm$est, 4), " se =", round(res_tm$se, 4), "\n\n")

# --- Method 2: SAR-EM (requires Python) ---
cat("[SAR-EM] Initializing Python...\n")
tryCatch({
  init_sarem(SAREM_VENV_PATH, SAREM_SARPU_PATH)
  res_sarem <- ate_onesample(X, S, Y, pu_method = "sarem", max_its = 500)
  cat("[SAR-EM] est =", round(res_sarem$est, 4),
      " se =", round(res_sarem$se, 4), "\n\n")
}, error = function(e) {
  cat("[SAR-EM] SKIPPED:", e$message, "\n\n")
})

# --- Method 3: DETM (requires PUEM) ---
cat("[DETM] Fitting via wrapper...\n")
tryCatch({
  library(PUEM)
  res_detm <- ate_onesample(X, S, Y, pu_method = "detm")
  cat("[DETM] est =", round(res_detm$est, 4),
      " se =", round(res_detm$se, 4), "\n\n")
}, error = function(e) {
  cat("[DETM] SKIPPED:", e$message, "\n\n")
})
