# example_twosample.R
# Two-sample (case-control) ATE estimation using PU learning
#
# DGP:
#   Pop 0 (S=0): n0=1000, A ~ Bern(0.3), X|A=1 ~ MVN(0,0), X|A=0 ~ MVN(0.5,-0.5)
#   Pop 1 (S=1): n1=500,  A=1 (all treated), X ~ MVN(m_shift, -m_shift)
#   Y^0 = X1 - X2 + eps;  Y^1 = Y^0 + 0.5 + 0.25*X1 - 0.25*X2
#   True ATE (Pop 0) = 0.675

library(MASS)
source("R/causalPUtrt.R")

# --- Parameters ---
n0      <- 1000
n1      <- 500
m_shift <- 0.4
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
ATE_true <- 0.5 + 0.25*0.3*(0 - 0.5) + 0.25*0.7*(0.5 - (-0.5)) +
            (-0.25)*0.3*(0 - (-0.5)) + (-0.25)*0.7*((-0.5) - 0.5)
# Simplified: compute from marginal moments of Pop 0
mu_X_pop0 <- 0.3 * c(0, 0) + 0.7 * c(0.5, -0.5)
ATE_true  <- 0.5 + 0.25 * mu_X_pop0[1] - 0.25 * mu_X_pop0[2]

cat("True ATE (Pop 0):", ATE_true, "\n")
cat("n0 =", n0, ", n1 =", n1, "\n\n")

# --- Sample splitting on Pop 0 + Pop 1 combined ---
split_idx <- sample(1:n, n/2)

X1_split <- X[split_idx, ]
S1_split <- S[split_idx]
Y1_split <- Y[split_idx]

# --- Step 1: Estimate piA using TM on Set 1 ---
cat("Fitting TM...\n")
tm_fit <- pu_learn_tm(X1_split, S1_split, epochs = 500)
piA_hat <- predict_proba_pu(tm_fit$model_clf, X)
piA_hat <- pmax(0.001, pmin(0.999, piA_hat))

# --- Step 2: Estimate nuisance parameters on Set 1 ---
# piS = P(S=1|X)
fit_piS <- glm(S ~ X1 + X2,
               data = data.frame(S = S1_split, X1 = X1_split[,1], X2 = X1_split[,2]),
               family = binomial())
piS_hat <- predict(fit_piS, newdata = data.frame(X1 = X[,1], X2 = X[,2]),
                   type = "response")

# mu = E[Y|X] (marginal outcome model)
fit_mu <- lm(Y ~ X1 + X2,
             data = data.frame(Y = Y1_split, X1 = X1_split[,1], X2 = X1_split[,2]))
mu_hat <- predict(fit_mu, newdata = data.frame(X1 = X[,1], X2 = X[,2]))

# mu1 = E[Y|S=1, X]
S1_labeled <- which(S1_split == 1)
fit_mu1 <- lm(Y ~ X1 + X2,
              data = data.frame(Y = Y1_split[S1_labeled],
                                X1 = X1_split[S1_labeled, 1],
                                X2 = X1_split[S1_labeled, 2]))
mu1_hat <- predict(fit_mu1, newdata = data.frame(X1 = X[,1], X2 = X[,2]))

# --- Step 3: Compute ATE estimates ---
cat("\n=== ATE Results (True ATE =", ATE_true, ") ===\n\n")

dat <- data.frame(S = S, Y = Y)

res_proposed <- proposed_estimator_twosample(dat, piA_hat, piS_hat, mu_hat, mu1_hat)
cat("Proposed (TM):  est =", round(res_proposed["est"], 4),
    " se =", round(res_proposed["se"], 4),
    " 95% CI = [", round(res_proposed["ci_lower"], 4), ",",
    round(res_proposed["ci_upper"], 4), "]\n")

res_naive <- naive_estimator_twosample(dat)
cat("Naive:          est =", round(res_naive["est"], 4),
    " se =", round(res_naive["se"], 4),
    " 95% CI = [", round(res_naive["ci_lower"], 4), ",",
    round(res_naive["ci_upper"], 4), "]\n")
