# ate_estimators.R
# Doubly robust ATE estimators for PU treatment settings.
#
# Two settings:
#   1. One-sample:  All units from one population. Some treated (A=1),
#                   some not (A=0). Label S=1 iff A=1 and selected.
#   2. Two-sample:  Pop 0 (S=0) and Pop 1 (S=1) are sampled separately
#                   (case-control design).

# ==============================================================================
# One-Sample Estimators
# ==============================================================================

#' Proposed DR estimator (one-sample, with sample splitting)
#'
#' Score: (mu1 + S*(Y - mu1)/piS - Y) / (1 - piA)
#'
#' @param S PU labels (0/1), full data
#' @param Y Outcome, full data
#' @param piA_hat Estimated P(A=1|X), full data (from PU learning on Set 1)
#' @param piS_hat Estimated P(S=1|X), full data (fitted on Set 1)
#' @param split_idx Indices of Set 1 (used for fitting; excluded from estimation)
#' @param mu1_hat Estimated E[Y|S=1,X], full data (fitted on Set 1)
#' @param trunc Truncation bounds for probabilities (default: c(0.001, 0.999))
#' @return Named vector: est, var, se, ci_lower, ci_upper
#' @export
proposed_estimator_onesample <- function(S, Y, piA_hat, piS_hat, split_idx,
                                         mu1_hat, trunc = c(0.001, 0.999)) {
  # Use only Set 2 (estimation set)
  S2     <- S[-split_idx]
  Y2     <- Y[-split_idx]
  piA2   <- pmax(trunc[1], pmin(trunc[2], piA_hat[-split_idx]))
  piS2   <- pmax(trunc[1], pmin(trunc[2], piS_hat[-split_idx]))
  mu1_2  <- mu1_hat[-split_idx]

  scores <- (mu1_2 + S2 * (Y2 - mu1_2) / piS2 - Y2) / (1 - piA2)

  est      <- mean(scores)
  n2       <- length(S2)
  var_est  <- var(scores)
  se       <- sqrt(var_est / n2)
  ci_lower <- est - 1.96 * se
  ci_upper <- est + 1.96 * se

  return(c(est = est, var = var_est, se = se,
           ci_lower = ci_lower, ci_upper = ci_upper))
}

#' Naive DR estimator (one-sample, with sample splitting)
#'
#' Treats S as if it were A. DR score using piS as propensity.
#' Score: (mu1 + S*(Y-mu1)/piS) - (mu0 + (1-S)*(Y-mu0)/(1-piS))
#'
#' @param S PU labels (0/1), full data
#' @param Y Outcome, full data
#' @param piS_hat Estimated P(S=1|X), full data
#' @param split_idx Indices of Set 1
#' @param mu1_hat Estimated E[Y|S=1,X], full data
#' @param mu0_hat Estimated E[Y|S=0,X], full data
#' @param trunc Truncation bounds (default: c(0.001, 0.999))
#' @return Named vector: est, var, se, ci_lower, ci_upper
#' @export
naive_estimator_onesample <- function(S, Y, piS_hat, split_idx,
                                      mu1_hat, mu0_hat,
                                      trunc = c(0.001, 0.999)) {
  S2     <- S[-split_idx]
  Y2     <- Y[-split_idx]
  piS2   <- pmax(trunc[1], pmin(trunc[2], piS_hat[-split_idx]))
  mu1_2  <- mu1_hat[-split_idx]
  mu0_2  <- mu0_hat[-split_idx]

  term1  <- mu1_2 + S2 * (Y2 - mu1_2) / piS2
  term0  <- mu0_2 + (1 - S2) * (Y2 - mu0_2) / (1 - piS2)
  scores <- term1 - term0

  est      <- mean(scores)
  n2       <- length(S2)
  var_est  <- var(scores)
  se       <- sqrt(var_est / n2)
  ci_lower <- est - 1.96 * se
  ci_upper <- est + 1.96 * se

  return(c(est = est, var = var_est, se = se,
           ci_lower = ci_lower, ci_upper = ci_upper))
}


# ==============================================================================
# Two-Sample Estimators
# ==============================================================================

#' Proposed DR estimator (two-sample / case-control, with sample splitting)
#'
#' For data where Pop 0 (S=0) and Pop 1 (S=1) are sampled separately.
#'
#' @param S PU labels (0/1), full data
#' @param Y Outcome, full data
#' @param piA_hat Estimated P(A=1|X), full data
#' @param piS_hat Estimated P(S=1|X), full data
#' @param mu_hat  Estimated E[Y|X], full data
#' @param mu1_hat Estimated E[Y|S=1,X], full data
#' @param split_idx Indices of Set 1 (used for fitting; excluded from estimation)
#' @param trunc Truncation bounds (default: c(0.001, 0.999))
#' @return Named vector: est, var, se, ci_lower, ci_upper
#' @export
proposed_estimator_twosample <- function(S, Y, piA_hat, piS_hat, mu_hat, mu1_hat,
                                         split_idx, trunc = c(0.001, 0.999)) {
  # Use only Set 2 (estimation set)
  S2     <- S[-split_idx]
  Y2     <- Y[-split_idx]
  piA2   <- pmax(trunc[1], pmin(trunc[2], piA_hat[-split_idx]))
  piS2   <- pmax(trunc[1], pmin(trunc[2], piS_hat[-split_idx]))
  mu2    <- mu_hat[-split_idx]
  mu1_2  <- mu1_hat[-split_idx]

  n2 <- length(S2)
  n0 <- sum(S2 == 0)

  ghat <- (1 - piS2) / (1 - piA2)

  scores <- ghat * (S2 - piS2) * (Y2 - mu1_2) / piS2 +
    (S2 - piS2) * (mu2 - mu1_2) / (1 - piA2)

  est     <- sum(scores) / n0
  var_est <- mean(scores^2) / (n0 / n2)^2
  se      <- sqrt(var_est / n2)

  ci_lower <- est - 1.96 * se
  ci_upper <- est + 1.96 * se

  return(c(est = est, var = var_est, se = se,
           ci_lower = ci_lower, ci_upper = ci_upper))
}

#' Naive DR estimator (two-sample, with sample splitting)
#'
#' DR score: (mu1 + S*(Y-mu1)/piS) - (mu0 + (1-S)*(Y-mu0)/(1-piS))
#'
#' @param S PU labels (0/1), full data
#' @param Y Outcome, full data
#' @param piS_hat Estimated P(S=1|X), full data
#' @param split_idx Indices of Set 1
#' @param mu1_hat Estimated E[Y|S=1,X], full data
#' @param mu0_hat Estimated E[Y|S=0,X], full data
#' @param trunc Truncation bounds (default: c(0.001, 0.999))
#' @return Named vector: est, var, se, ci_lower, ci_upper
#' @export
naive_estimator_twosample <- function(S, Y, piS_hat, split_idx,
                                      mu1_hat, mu0_hat,
                                      trunc = c(0.001, 0.999)) {
  S2     <- S[-split_idx]
  Y2     <- Y[-split_idx]
  piS2   <- pmax(trunc[1], pmin(trunc[2], piS_hat[-split_idx]))
  mu1_2  <- mu1_hat[-split_idx]
  mu0_2  <- mu0_hat[-split_idx]

  term1  <- mu1_2 + S2 * (Y2 - mu1_2) / piS2
  term0  <- mu0_2 + (1 - S2) * (Y2 - mu0_2) / (1 - piS2)
  scores <- term1 - term0

  est      <- mean(scores)
  n2       <- length(S2)
  var_est  <- var(scores)
  se       <- sqrt(var_est / n2)
  ci_lower <- est - 1.96 * se
  ci_upper <- est + 1.96 * se

  return(c(est = est, var = var_est, se = se,
           ci_lower = ci_lower, ci_upper = ci_upper))
}
