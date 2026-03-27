# example_onesample.R
# One-sample ATE estimation using PU learning
#
# DGP:
#   X ~ MVN(mu=[1,-1], Sigma), n=1500
#   A|X ~ Bernoulli(logit^{-1}(X1 + X2))          # true treatment (beta0=0)
#   S|A=1,X ~ Bernoulli(logit^{-1}(0.8*(X1+X2)))  # labeling; S=0 if A=0
#   Y^0 = X1 - X2 + eps,  Y^1 = Y^0 + 0.5 + 0.25*X1 - 0.25*X2
#   True ATE = 1.0

library(MASS)
source("R/causalPUtrt.R")

# --- Parameters ---
n      <- 1500
beta0  <- 0         # P(A=1) ~ 0.5
beta   <- 1.0
gamma  <- 0.8       # SAR labeling strength
tau    <- 1.0

# --- Generate data ---
set.seed(42)
mu_X    <- c(1, -1)
Sigma_X <- matrix(c(1, 0.2, 0.2, 1), 2, 2)
X       <- mvrnorm(n, mu_X, Sigma_X)
colnames(X) <- c("X1", "X2")

piA_true  <- plogis(beta0 + beta * X[,1] + beta * X[,2])
A         <- rbinom(n, 1, piA_true)
e_true    <- plogis(gamma * X[,1] + gamma * X[,2])
S         <- A * rbinom(n, 1, e_true)
Y0        <- X[,1] - X[,2] + rnorm(n)
Y1        <- Y0 + tau/2 + (tau/4)*X[,1] - (tau/4)*X[,2]
Y         <- ifelse(A == 0, Y0, Y1)
ATE_true  <- tau/2 + mu_X[1]*(tau/4) + mu_X[2]*(-tau/4)

cat("True ATE:", ATE_true, "\n")
cat("Observed: n =", n, ", sum(S=1) =", sum(S), ", P(A=1) =", round(mean(A), 3), "\n\n")

# ===========================================================
# Method 1: SAR-EM via wrapper (requires Python sarpu)
# ===========================================================
cat("=== SAR-EM via wrapper ===\n\n")

tryCatch({
  # init_sarem("/path/to/venv", "/path/to/SAR-PU/sarpu")
  # res_sarem <- ate_onesample(X, S, Y, pu_method = "sarem")
  # cat("SAR-EM:  est =", round(res_sarem$est, 4),
  #     " se =", round(res_sarem$se, 4),
  #     " 95% CI = [", round(res_sarem$ci_lower, 4), ",",
  #     round(res_sarem$ci_upper, 4), "]\n\n")
  cat("(Skipped: uncomment and set paths to run SAR-EM)\n\n")
}, error = function(e) {
  cat("SAR-EM not available:", e$message, "\n\n")
})

# ===========================================================
# Method 2: Custom piA_hat (any external PU method)
# ===========================================================
cat("=== Custom piA_hat ===\n\n")

# Example: use a simple logistic regression on S as a rough piA estimate
fit_rough <- glm(S ~ X1 + X2,
                 data = data.frame(S = S, X1 = X[,1], X2 = X[,2]),
                 family = binomial())
piA_custom <- predict(fit_rough, type = "response")
# In practice, use your preferred PU method to get piA_hat

res_custom <- ate_onesample(X, S, Y, piA_hat = piA_custom)
cat("Custom piA: est =", round(res_custom$est, 4),
    " se =", round(res_custom$se, 4),
    " 95% CI = [", round(res_custom$ci_lower, 4), ",",
    round(res_custom$ci_upper, 4), "]\n\n")

# ===========================================================
# Method 3: Low-level estimators (for custom pipelines)
# ===========================================================
cat("=== Low-level estimators ===\n\n")

# Sample splitting
split_idx <- sample(1:n, n/2)
X1 <- X[split_idx, ]
S1 <- S[split_idx]
Y1_split <- Y[split_idx]

# Use custom piA (e.g., from logistic regression on S)
fit_piA <- glm(S ~ X1 + X2,
               data = data.frame(S = S1, X1 = X1[,1], X2 = X1[,2]),
               family = binomial())
piA_hat <- predict(fit_piA, newdata = data.frame(X1 = X[,1], X2 = X[,2]),
                   type = "response")

# Nuisance parameters on Set 1
fit_piS <- glm(S ~ X1 + X2,
               data = data.frame(S = S1, X1 = X1[,1], X2 = X1[,2]),
               family = binomial())
piS_hat <- predict(fit_piS, newdata = data.frame(X1 = X[,1], X2 = X[,2]),
                   type = "response")

S1_labeled <- which(S1 == 1)
fit_mu1 <- lm(Y ~ X1 + X2,
              data = data.frame(Y = Y1_split[S1_labeled],
                                X1 = X1[S1_labeled, 1],
                                X2 = X1[S1_labeled, 2]))
mu1_hat <- predict(fit_mu1, newdata = data.frame(X1 = X[,1], X2 = X[,2]))

S1_unlabeled <- which(S1 == 0)
fit_mu0 <- lm(Y ~ X1 + X2,
              data = data.frame(Y = Y1_split[S1_unlabeled],
                                X1 = X1[S1_unlabeled, 1],
                                X2 = X1[S1_unlabeled, 2]))
mu0_hat <- predict(fit_mu0, newdata = data.frame(X1 = X[,1], X2 = X[,2]))

# ATE estimates
cat("ATE Results (True ATE =", ATE_true, "):\n\n")

res_proposed <- proposed_estimator_onesample(S, Y, piA_hat, piS_hat,
                                             split_idx, mu1_hat)
cat("Proposed:  est =", round(res_proposed["est"], 4),
    " se =", round(res_proposed["se"], 4),
    " 95% CI = [", round(res_proposed["ci_lower"], 4), ",",
    round(res_proposed["ci_upper"], 4), "]\n")

res_naive <- naive_estimator_onesample(S, Y, piS_hat, split_idx,
                                       mu1_hat, mu0_hat)
cat("Naive:     est =", round(res_naive["est"], 4),
    " se =", round(res_naive["se"], 4),
    " 95% CI = [", round(res_naive["ci_lower"], 4), ",",
    round(res_naive["ci_upper"], 4), "]\n")
