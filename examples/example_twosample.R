# example_twosample.R
# Two-sample (case-control) ATE estimation using PU learning
#
# DGP:
#   Pop 0 (S=0): n0=1000, A ~ Bern(0.3), X|A=1 ~ MVN(0,0), X|A=0 ~ MVN(0.5,-0.5)
#   Pop 1 (S=1): n1=500,  A=1 (all treated), X ~ MVN(m, -m), m=0.2
#   Y^0 = X1 - X2 + eps;  Y^1 = Y^0 + 0.5 + 0.25*X1 - 0.25*X2
# Target: tau^UL = E(Y^1 - Y^0 | S=0)

library(MASS)
source("R/causalPUtrt.R")

# --- Parameters ---
n0      <- 1000
n1      <- 500
m_shift <- 0.2
pA_pop0 <- 0.3

# --- Generate data ---
set.seed(123)
Sigma <- matrix(c(1, 0.2, 0.2, 1), 2, 2)

# Pop 0 (S=0)
A0 <- rbinom(n0, 1, pA_pop0)
X0 <- matrix(NA, n0, 2)
X0[A0 == 1, ] <- mvrnorm(sum(A0), c(0, 0), Sigma)
X0[A0 == 0, ] <- mvrnorm(n0 - sum(A0), c(0.5, -0.5), Sigma)

# Pop 1 (S=1): all treated
X1 <- mvrnorm(n1, c(m_shift, -m_shift), Sigma)
A1 <- rep(1, n1)

# Combine
X <- rbind(X0, X1)
colnames(X) <- c("X1", "X2")
A <- c(A0, A1)
S <- c(rep(0, n0), rep(1, n1))
n <- n0 + n1

# Outcomes
Y0_pot <- X[,1] - X[,2] + rnorm(n)
Y1_pot <- Y0_pot + 0.5 + 0.25*X[,1] - 0.25*X[,2]
Y <- ifelse(A == 0, Y0_pot, Y1_pot)

# True ATE for Pop 0
mu_X_pop0 <- 0.3 * c(0, 0) + 0.7 * c(0.5, -0.5)
ATE_true  <- 0.5 + 0.25 * mu_X_pop0[1] - 0.25 * mu_X_pop0[2]

cat("True ATE (Pop 0):", ATE_true, "\n")
cat("n0 =", n0, ", n1 =", n1, "\n\n")

# ===========================================================
# Method 1: SCAR via wrapper (pure R)
# ===========================================================
cat("=== SCAR via wrapper ===\n\n")

# SCAR requires the labeling rate c = P(S=1|A=1) to be KNOWN. In the combined
# case-control sample, the treated units are Pop 1 (all labeled) plus the
# treated part of Pop 0 (unlabeled), so the oracle value is
# c = n1 / (n1 + #treated in Pop 0). In practice c must come from external
# knowledge (it is not identifiable from (X, S, Y) alone).
c_known <- n1 / (n1 + sum(A0))
cat("c_known (oracle):", round(c_known, 4), "\n")

res_scar <- ate_twosample(X, S, Y, pu_method = "scar", c_known = c_known)
cat("SCAR:    est =", round(res_scar$est, 4),
    " se =", round(res_scar$se, 4),
    " 95% CI = [", round(res_scar$ci_lower, 4), ",",
    round(res_scar$ci_upper, 4), "]\n\n")

# ===========================================================
# Method 2: SAR-EM via wrapper (requires Python sarpu)
# ===========================================================
cat("=== SAR-EM via wrapper ===\n\n")

tryCatch({
  # init_sarem("/path/to/venv", "/path/to/SAR-PU/sarpu")
  # res_sarem <- ate_twosample(X, S, Y, pu_method = "sarem")
  # cat("SAR-EM:  est =", round(res_sarem$est, 4),
  #     " se =", round(res_sarem$se, 4),
  #     " 95% CI = [", round(res_sarem$ci_lower, 4), ",",
  #     round(res_sarem$ci_upper, 4), "]\n\n")
  cat("(Skipped: uncomment and set paths to run SAR-EM)\n\n")
}, error = function(e) {
  cat("SAR-EM not available:", e$message, "\n\n")
})

# ===========================================================
# Method 3: Custom piA_hat (any external source)
# ===========================================================
cat("=== Custom piA_hat ===\n\n")

# Example: use a simple logistic regression on S as a rough piA estimate.
# NOTE: with a user-supplied piA_hat the theta-correction is omitted; the
# reported SE is valid only if piA is known or estimated at o_p(n^{-1/2})
# rate (condition (R1) of Theorem 2).
fit_rough <- glm(S ~ X1 + X2,
                 data = data.frame(S = S, X1 = X[,1], X2 = X[,2]),
                 family = binomial())
piA_custom <- predict(fit_rough, type = "response")

res_custom <- ate_twosample(X, S, Y, piA_hat = piA_custom)
cat("Custom piA: est =", round(res_custom$est, 4),
    " se =", round(res_custom$se, 4),
    " 95% CI = [", round(res_custom$ci_lower, 4), ",",
    round(res_custom$ci_upper, 4), "]\n\n")

# ===========================================================
# Method 4: Low-level estimators (for custom pipelines)
# ===========================================================
cat("=== Low-level estimators ===\n\n")

split_idx <- sample(1:n, n/2)
X1_split <- X[split_idx, ]
S1_split <- S[split_idx]
Y1_split <- Y[split_idx]
ev <- setdiff(1:n, split_idx)

# piA via SCAR MLE (pure R) with its influence function
Z_tr <- cbind(1, X1_split)
Z_ev <- cbind(1, X[ev, , drop = FALSE])
fs   <- fit_scar_mle(Z_tr, S1_split, c_known)
piA_hat <- as.vector(plogis(cbind(1, X) %*% fs$theta))
Xi_ev   <- xi_scar_eval(fs$theta, c_known, Z_tr, S1_split, Z_ev, S[ev])

# Nuisance
fit_piS <- glm(S ~ X1 + X2,
               data = data.frame(S = S1_split, X1 = X1_split[,1], X2 = X1_split[,2]),
               family = binomial())
piS_hat <- predict(fit_piS, newdata = data.frame(X1 = X[,1], X2 = X[,2]),
                   type = "response")

fit_mu <- lm(Y ~ X1 + X2,
             data = data.frame(Y = Y1_split, X1 = X1_split[,1], X2 = X1_split[,2]))
mu_hat <- predict(fit_mu, newdata = data.frame(X1 = X[,1], X2 = X[,2]))

S1_labeled <- which(S1_split == 1)
fit_mu1 <- lm(Y ~ X1 + X2,
              data = data.frame(Y = Y1_split[S1_labeled],
                                X1 = X1_split[S1_labeled, 1],
                                X2 = X1_split[S1_labeled, 2]))
mu1_hat <- predict(fit_mu1, newdata = data.frame(X1 = X[,1], X2 = X[,2]))

# mu0 for naive DR
S1_unlabeled <- which(S1_split == 0)
fit_mu0 <- lm(Y ~ X1 + X2,
              data = data.frame(Y = Y1_split[S1_unlabeled],
                                X1 = X1_split[S1_unlabeled, 1],
                                X2 = X1_split[S1_unlabeled, 2]))
mu0_hat <- predict(fit_mu0, newdata = data.frame(X1 = X[,1], X2 = X[,2]))

cat("ATE Results (True ATE =", ATE_true, "):\n\n")

# With the theta-correction (Z_ev + Xi_ev supplied)
res_proposed <- proposed_estimator_twosample(S, Y, piA_hat, piS_hat,
                                             mu_hat, mu1_hat, split_idx,
                                             Z_ev = Z_ev, Xi_ev = Xi_ev)
cat("Proposed (corrected SE):   est =", round(res_proposed["est"], 4),
    " se =", round(res_proposed["se"], 4),
    " 95% CI = [", round(res_proposed["ci_lower"], 4), ",",
    round(res_proposed["ci_upper"], 4), "]\n")

# Without the correction (omit Xi_ev; only valid if piA is known or
# estimated at o_p(n^{-1/2}) rate)
res_nocorr <- proposed_estimator_twosample(S, Y, piA_hat, piS_hat,
                                           mu_hat, mu1_hat, split_idx)
cat("Proposed (no correction):  est =", round(res_nocorr["est"], 4),
    " se =", round(res_nocorr["se"], 4), "\n")

res_naive <- naive_estimator_twosample(S, Y, piS_hat, split_idx,
                                       mu1_hat, mu0_hat)
cat("Naive DR:                  est =", round(res_naive["est"], 4),
    " se =", round(res_naive["se"], 4),
    " 95% CI = [", round(res_naive["ci_lower"], 4), ",",
    round(res_naive["ci_upper"], 4), "]\n")
