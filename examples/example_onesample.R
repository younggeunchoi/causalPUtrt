# example_onesample.R
# One-sample ATE estimation using PU learning
#
# DGP:
#   X ~ MVN(mu=[1,-1], Sigma), n=1500
#   A|X ~ Bernoulli(logit^{-1}(beta0 + X1 + X2))    # true treatment
#   S|A=1,X ~ Bernoulli(logit^{-1}(gamma*(X1+X2)))   # labeling; S=0 if A=0
#   Y^0 = X1 - X2 + eps,  Y^1 = Y^0 + 0.5 + 0.25*X1 - 0.25*X2
#   True ATE = 1.0

library(MASS)
source("R/causalPUtrt.R")

# --- Parameters ---
n      <- 1500
beta0  <- -1.0986   # P(A=1) ~ 0.25
beta   <- 1.0
gamma  <- 0.4       # SAR labeling strength
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

# --- Sample splitting ---
split_idx <- sample(1:n, n/2)
X1 <- X[split_idx, ]
S1 <- S[split_idx]
Y1_split <- Y[split_idx]

# --- Step 1: Estimate piA using TM (pure R, no external deps) ---
cat("Fitting TM...\n")
tm_fit <- pu_learn_tm(X1, S1, epochs = 500)
piA_tm <- predict_proba_pu(tm_fit$model_clf, X)

# --- Step 2: Estimate nuisance parameters on Set 1 ---
# piS = P(S=1|X)
fit_piS <- glm(S ~ X1 + X2,
               data = data.frame(S = S1, X1 = X1[,1], X2 = X1[,2]),
               family = binomial())
piS_hat <- predict(fit_piS, newdata = data.frame(X1 = X[,1], X2 = X[,2]),
                   type = "response")

# mu1 = E[Y|S=1,X]
S1_labeled <- which(S1 == 1)
fit_mu1 <- lm(Y ~ X1 + X2,
              data = data.frame(Y = Y1_split[S1_labeled],
                                X1 = X1[S1_labeled, 1],
                                X2 = X1[S1_labeled, 2]))
mu1_hat <- predict(fit_mu1, newdata = data.frame(X1 = X[,1], X2 = X[,2]))

# mu0 = E[Y|S=0,X] (for naive and Kato)
S1_unlabeled <- which(S1 == 0)
fit_mu0 <- lm(Y ~ X1 + X2,
              data = data.frame(Y = Y1_split[S1_unlabeled],
                                X1 = X1[S1_unlabeled, 1],
                                X2 = X1[S1_unlabeled, 2]))
mu0_hat <- predict(fit_mu0, newdata = data.frame(X1 = X[,1], X2 = X[,2]))

# --- Step 3: Compute ATE estimates ---
cat("\n=== ATE Results (True ATE =", ATE_true, ") ===\n\n")

res_proposed <- proposed_estimator_onesample(X, S, Y, piA_tm, piS_hat,
                                             split_idx, mu1_hat)
cat("Proposed (TM):  est =", round(res_proposed["est"], 4),
    " se =", round(res_proposed["se"], 4),
    " 95% CI = [", round(res_proposed["ci_lower"], 4), ",",
    round(res_proposed["ci_upper"], 4), "]\n")

res_naive <- naive_estimator_onesample(X, S, Y, piS_hat, split_idx,
                                       mu1_hat, mu0_hat)
cat("Naive:          est =", round(res_naive["est"], 4),
    " se =", round(res_naive["se"], 4),
    " 95% CI = [", round(res_naive["ci_lower"], 4), ",",
    round(res_naive["ci_upper"], 4), "]\n")

res_kato <- kato_estimator_onesample(X, S, Y, piA_tm, piS_hat, split_idx,
                                     mu1_hat, mu0_hat)
cat("Kato (TM):      est =", round(res_kato["est"], 4),
    " se =", round(res_kato["se"], 4),
    " 95% CI = [", round(res_kato["ci_lower"], 4), ",",
    round(res_kato["ci_upper"], 4), "]\n")
