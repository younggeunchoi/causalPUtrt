# ate_estimators.R
# Doubly robust ATE estimators for PU treatment settings.
#
# Two settings:
#   1. One-sample:  All units from one population. Some treated (A=1),
#                   some not (A=0). Label S=1 iff A=1 and selected.
#   2. Two-sample:  Pop 0 (S=0) and Pop 1 (S=1) are sampled separately
#                   (case-control design).
#
# Variance: the corrected (Theorem 3 / Theorem S3) standard error, i.e., the
# empirical variance over the eval half of the per-observation partial
# influence function psi_par_i = psi0_i + B' xi_theta_i, where xi_theta is
# the influence function of theta-hat (the parametric piA coefficients).
# xi_theta is METHOD-DEPENDENT and provided only for SAR-EM (xi_sar_eval)
# and SCAR (xi_scar_eval); see README. If Xi_ev is omitted, the correction
# term is zero, which is valid only when piA is known or estimated at
# o_p(n^{-1/2}) rate (condition (R1) of Theorem 2).

# ==============================================================================
# One-Sample Estimators
# ==============================================================================

#' Proposed AIPW estimator (one-sample, with sample splitting, corrected SE)
#'
#' Score: D / (1 - piA), with D = mu1 + S*(Y - mu1)/piS - Y.
#' SE (Theorem 3): sqrt(var(psi0 + Xi_ev %*% B_hat) / n_ev) on the eval half,
#' with B_hat = colMeans(Z_ev * (piA/(1-piA) * D)).
#'
#' @param S PU labels (0/1), full data
#' @param Y Outcome, full data
#' @param piA_hat Estimated P(A=1|X), full data (from PU learning on Set 1)
#' @param piS_hat Estimated P(S=1|X), full data (fitted on Set 1)
#' @param split_idx Indices of Set 1 (used for fitting; excluded from estimation)
#' @param mu1_hat Estimated E[Y|S=1,X], full data (fitted on Set 1)
#' @param Z_ev Design matrix (with intercept column) at the eval observations,
#'   i.e., rows setdiff(1:n, split_idx) in ascending order. Required when
#'   Xi_ev is supplied.
#' @param Xi_ev n_ev x r matrix of xi_theta values at the eval observations
#'   (from xi_sar_eval or xi_scar_eval). NULL = no theta-correction, valid
#'   only if piA is known or estimated at o_p(n^{-1/2}) rate.
#' @param trunc Truncation bounds for probabilities (default: c(0.001, 0.999))
#' @return Named vector: est, se, ci_lower, ci_upper
#' @export
proposed_estimator_onesample <- function(S, Y, piA_hat, piS_hat, split_idx,
                                         mu1_hat, Z_ev = NULL, Xi_ev = NULL,
                                         trunc = c(0.001, 0.999)) {
  # Use only Set 2 (estimation set); [-split_idx] keeps ascending order,
  # matching Z_ev/Xi_ev rows built from setdiff(1:n, split_idx)
  S2     <- S[-split_idx]
  Y2     <- Y[-split_idx]
  piA2   <- pmax(trunc[1], pmin(trunc[2], piA_hat[-split_idx]))
  piS2   <- pmax(trunc[1], pmin(trunc[2], piS_hat[-split_idx]))
  mu1_2  <- mu1_hat[-split_idx]

  n_ev   <- length(S2)
  D_ev   <- mu1_2 + S2 * (Y2 - mu1_2) / piS2 - Y2
  scores <- D_ev / (1 - piA2)
  est    <- mean(scores)
  psi0   <- scores - est

  if (!is.null(Xi_ev)) {
    if (is.null(Z_ev)) stop("Z_ev must be supplied together with Xi_ev.")
    B_hat   <- colMeans(Z_ev * (piA2 / (1 - piA2) * D_ev))
    psi_par <- psi0 + as.vector(Xi_ev %*% B_hat)
  } else {
    psi_par <- psi0
  }

  se       <- sqrt(var(psi_par) / n_ev)
  ci_lower <- est - 1.96 * se
  ci_upper <- est + 1.96 * se

  return(c(est = est, se = se, ci_lower = ci_lower, ci_upper = ci_upper))
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
#' @return Named vector: est, se, ci_lower, ci_upper
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
  se       <- sqrt(var(scores) / n2)
  ci_lower <- est - 1.96 * se
  ci_upper <- est + 1.96 * se

  return(c(est = est, se = se, ci_lower = ci_lower, ci_upper = ci_upper))
}


# ==============================================================================
# Two-Sample Estimators
# ==============================================================================

#' Proposed AIPW estimator (two-sample / case-control, corrected SE)
#'
#' For data where Pop 0 (S=0) and Pop 1 (S=1) are sampled separately.
#' Targets tau^UL = E(Y^1 - Y^0 | S=0). The ATE average and p0 are computed
#' on the eval half only.
#'
#' SE (Theorem S3): sqrt(var(psi0 + Xi_ev %*% B_UL) / n_ev), with
#'   psi0 = scores/p0 - est*(1-S)/p0,
#'   B_UL = colMeans(Z_ev * ((1-piS) * piA/(1-piA) * (mu1 - mu))) / p0.
#'
#' @param S PU labels (0/1), full data
#' @param Y Outcome, full data
#' @param piA_hat Estimated P(A=1|X), full data
#' @param piS_hat Estimated P(S=1|X), full data
#' @param mu_hat  Estimated E[Y|X], full data
#' @param mu1_hat Estimated E[Y|S=1,X], full data
#' @param split_idx Indices of Set 1 (used for fitting; excluded from estimation)
#' @param Z_ev Design matrix (with intercept column) at the eval observations.
#'   Required when Xi_ev is supplied.
#' @param Xi_ev n_ev x r matrix of xi_theta values at the eval observations
#'   (from xi_sar_eval or xi_scar_eval). NULL = no theta-correction, valid
#'   only if piA is known or estimated at o_p(n^{-1/2}) rate.
#' @param trunc Truncation bounds (default: c(0.001, 0.999))
#' @return Named vector: est, se, ci_lower, ci_upper
#' @export
proposed_estimator_twosample <- function(S, Y, piA_hat, piS_hat, mu_hat, mu1_hat,
                                         split_idx, Z_ev = NULL, Xi_ev = NULL,
                                         trunc = c(0.001, 0.999)) {
  # Use only Set 2 (estimation set)
  S2     <- S[-split_idx]
  Y2     <- Y[-split_idx]
  piA2   <- pmax(trunc[1], pmin(trunc[2], piA_hat[-split_idx]))
  piS2   <- pmax(trunc[1], pmin(trunc[2], piS_hat[-split_idx]))
  mu2    <- mu_hat[-split_idx]
  mu1_2  <- mu1_hat[-split_idx]

  n_ev   <- length(S2)
  n0_ev  <- sum(S2 == 0)
  p0     <- n0_ev / n_ev

  ghat   <- (1 - piS2) / (1 - piA2)
  scores <- ghat * (S2 - piS2) * (Y2 - mu1_2) / piS2 +
    (S2 - piS2) * (mu2 - mu1_2) / (1 - piA2)
  est    <- sum(scores) / n0_ev

  # psi^UL plug-in (Theorem S3)
  psi0   <- scores / p0 - est * (1 - S2) / p0

  if (!is.null(Xi_ev)) {
    if (is.null(Z_ev)) stop("Z_ev must be supplied together with Xi_ev.")
    B_UL    <- colMeans(Z_ev * ((1 - piS2) * (piA2 / (1 - piA2)) * (mu1_2 - mu2))) / p0
    psi_par <- psi0 + as.vector(Xi_ev %*% B_UL)
  } else {
    psi_par <- psi0
  }

  se       <- sqrt(var(psi_par) / n_ev)
  ci_lower <- est - 1.96 * se
  ci_upper <- est + 1.96 * se

  return(c(est = est, se = se, ci_lower = ci_lower, ci_upper = ci_upper))
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
#' @return Named vector: est, se, ci_lower, ci_upper
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
  se       <- sqrt(var(scores) / n2)
  ci_lower <- est - 1.96 * se
  ci_upper <- est + 1.96 * se

  return(c(est = est, se = se, ci_lower = ci_lower, ci_upper = ci_upper))
}
