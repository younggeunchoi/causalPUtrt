# example_all_methods.R
# Compare all three PU methods (SAR-EM, TM, DETM) for one-sample ATE
#
# NOTE: This example requires all three backends:
#   - TM:     pure R (always available)
#   - SAR-EM: Python sarpu package via reticulate
#   - DETM:   PUEM R package
# See README.md for installation instructions.

library(MASS)
source("R/causalPUtrt.R")

# --- CONFIGURE PATHS (edit these for your environment) ---
SAREM_VENV_PATH  <- "/path/to/your/virtualenv"       # Python virtualenv
SAREM_SARPU_PATH <- "/path/to/SAR_PU_python/sarpu"   # sarpu source directory

# --- Parameters ---
n      <- 1500
beta0  <- -1.0986
beta   <- 1.0
gamma  <- 0.4
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

# --- Sample splitting ---
split_idx <- sample(1:n, n/2)
X1 <- X[split_idx, ]
S1 <- S[split_idx]
Y1_data <- Y[split_idx]

# --- Nuisance parameters (correctly specified, fitted on Set 1) ---
fit_piS <- glm(S ~ X1 + X2,
               data = data.frame(S = S1, X1 = X1[,1], X2 = X1[,2]),
               family = binomial())
piS_hat <- predict(fit_piS, newdata = data.frame(X1 = X[,1], X2 = X[,2]),
                   type = "response")

S1_lab <- which(S1 == 1)
fit_mu1 <- lm(Y ~ X1 + X2,
              data = data.frame(Y = Y1_data[S1_lab],
                                X1 = X1[S1_lab, 1], X2 = X1[S1_lab, 2]))
mu1_hat <- predict(fit_mu1, newdata = data.frame(X1 = X[,1], X2 = X[,2]))


cat("=== Comparing PU Methods for ATE (True ATE =", ATE_true, ") ===\n\n")

# --- Method 1: TM (always available) ---
cat("[TM] Fitting...\n")
tm_fit  <- pu_learn_tm(X1, S1, epochs = 500)
piA_tm  <- predict_proba_pu(tm_fit$model_clf, X)
res_tm  <- proposed_estimator_onesample(X, S, Y, piA_tm, piS_hat, split_idx, mu1_hat)
cat("[TM] est =", round(res_tm["est"], 4), " se =", round(res_tm["se"], 4), "\n\n")

# --- Method 2: SAR-EM (requires Python) ---
cat("[SAR-EM] Initializing Python...\n")
tryCatch({
  init_sarem(SAREM_VENV_PATH, SAREM_SARPU_PATH)
  sarem_fit <- fit_sarem(X1, S1, max_its = 500)
  if (!sarem_fit$error) {
    preds     <- predict_sarem(sarem_fit, X)
    piA_sarem <- preds$prob_pred
    res_sarem <- proposed_estimator_onesample(X, S, Y, piA_sarem, piS_hat,
                                              split_idx, mu1_hat)
    cat("[SAR-EM] est =", round(res_sarem["est"], 4),
        " se =", round(res_sarem["se"], 4), "\n\n")
  } else {
    cat("[SAR-EM] FAILED to converge.\n\n")
  }
}, error = function(e) {
  cat("[SAR-EM] SKIPPED:", e$message, "\n\n")
})

# --- Method 3: DETM (requires PUEM) ---
cat("[DETM] Fitting...\n")
tryCatch({
  library(PUEM)
  detm_fit <- fit_detm(X1, S1)
  if (!detm_fit$error) {
    piA_detm <- predict_detm(detm_fit, X)
    res_detm <- proposed_estimator_onesample(X, S, Y, piA_detm, piS_hat,
                                             split_idx, mu1_hat)
    cat("[DETM] est =", round(res_detm["est"], 4),
        " se =", round(res_detm["se"], 4), "\n\n")
  } else {
    cat("[DETM] FAILED to converge.\n\n")
  }
}, error = function(e) {
  cat("[DETM] SKIPPED:", e$message, "\n\n")
})
