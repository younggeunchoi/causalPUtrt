# test_all_methods.R
# Clean environment test for SCAR, SAR-EM, and custom piA_hat
# Run from repo root via Docker (see Dockerfile).
# SAR-EM paths can be overridden with env vars SAREM_VENV / SARPU_PATH.

cat("=== Full clean install test (SCAR + SAR-EM + custom piA) ===\n\n")

source("R/causalPUtrt.R")

library(MASS)
set.seed(42)

# --- Generate data (same DGP as examples) ---
n <- 1500
mu_X <- c(1, -1)
Sigma_X <- matrix(c(1, 0.2, 0.2, 1), 2, 2)
X <- mvrnorm(n, mu_X, Sigma_X)
colnames(X) <- c("X1", "X2")

piA_true <- plogis(X[,1] + X[,2])
A <- rbinom(n, 1, piA_true)
e_true <- plogis(0.8 * (X[,1] + X[,2]))
S <- A * rbinom(n, 1, e_true)
Y0 <- X[,1] - X[,2] + rnorm(n)
Y1 <- Y0 + 0.5 + 0.25*X[,1] - 0.25*X[,2]
Y <- ifelse(A == 0, Y0, Y1)
ATE_true <- 1.0

cat("True ATE:", ATE_true, "\n")
cat("n =", n, ", sum(S=1) =", sum(S), "\n\n")

pass <- 0
fail <- 0

# --- SCAR (pure R, always available) ---
cat("--- [1/3] SCAR ---\n")
tryCatch({
  c_known <- mean(e_true[A == 1])   # oracle labeling rate (simulation only)
  res <- ate_onesample(X, S, Y, pu_method = "scar", c_known = c_known)
  cat("  est =", round(res$est, 4), " se =", round(res$se, 4), "\n  PASS\n\n")
  pass <- pass + 1
}, error = function(e) {
  cat("  FAIL:", e$message, "\n\n")
  fail <<- fail + 1
})

# --- SAR-EM ---
cat("--- [2/3] SAR-EM ---\n")
tryCatch({
  venv  <- Sys.getenv("SAREM_VENV", "/root/.virtualenvs/sarem_venv")
  sarpu <- Sys.getenv("SARPU_PATH", "/tmp/sarpu_src")
  init_sarem(venv, sarpu)
  res <- ate_onesample(X, S, Y, pu_method = "sarem", max_its = 500)
  cat("  est =", round(res$est, 4), " se =", round(res$se, 4), "\n  PASS\n\n")
  pass <- pass + 1
}, error = function(e) {
  cat("  FAIL:", e$message, "\n\n")
  fail <<- fail + 1
})

# --- Custom piA_hat (no theta-correction) ---
cat("--- [3/3] Custom piA_hat ---\n")
tryCatch({
  # Use logistic regression on S as a rough piA estimate
  fit_rough <- glm(S ~ X, family = binomial())
  piA_custom <- fitted(fit_rough)
  res <- ate_onesample(X, S, Y, piA_hat = piA_custom)
  cat("  est =", round(res$est, 4), " se =", round(res$se, 4), "\n  PASS\n\n")
  pass <- pass + 1
}, error = function(e) {
  cat("  FAIL:", e$message, "\n\n")
  fail <<- fail + 1
})

# --- Summary ---
cat("=== Results:", pass, "passed,", fail, "failed ===\n")
if (fail > 0) quit(status = 1)
