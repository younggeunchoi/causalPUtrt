# ate_wrappers.R
# End-to-end wrapper functions for ATE estimation with PU learning.
#
# These wrappers handle the full pipeline:
#   sample splitting -> piA estimation (SAR-EM / SCAR / user-supplied)
#   -> nuisance estimation -> ATE with the corrected (Theorem 3/S3) SE.

# Internal: shared piA-estimation step for both wrappers.
# Returns list(piA_hat, Xi_ev, theta, converged, pu_fit).
.estimate_piA <- function(X, S, pu_method, piA_hat, c_known,
                          split_idx, max_its, C) {
  n  <- nrow(X)
  X1 <- X[split_idx, , drop = FALSE]
  S1 <- S[split_idx]
  ev <- setdiff(1:n, split_idx)

  Z_tr <- cbind(1, X1)
  Z_ev <- cbind(1, X[ev, , drop = FALSE])
  S_ev <- S[ev]
  ridge <- 1 / (length(split_idx) * C)

  if (!is.null(piA_hat)) {
    # User-supplied piA estimates: no theta-correction (Xi_ev = NULL).
    # Valid only if piA is known or estimated at o_p(n^{-1/2}) rate.
    if (length(piA_hat) != n) {
      stop("piA_hat must have length n (", n, "), got ", length(piA_hat))
    }
    return(list(piA_hat = piA_hat, Xi_ev = NULL, Z_ev = Z_ev,
                theta = NULL, converged = NA, pu_fit = NULL))
  }

  if (pu_method == "sarem") {
    pu_fit <- fit_sarem(X1, S1, max_its = max_its, C = C)
    if (pu_fit$error)      stop("SAR-EM failed.")
    if (!pu_fit$converged) stop("SAR-EM did not converge.")
    theta  <- pu_fit$theta
    piA    <- predict_sarem(pu_fit, X)$prob_pred
    Xi_ev  <- xi_sar_eval(theta, pu_fit$phi, Z_tr, Z_tr, S1,
                          Z_ev, Z_ev, S_ev, ridge)
    return(list(piA_hat = piA, Xi_ev = Xi_ev, Z_ev = Z_ev,
                theta = theta, converged = pu_fit$converged, pu_fit = pu_fit))
  }

  if (pu_method == "scar") {
    if (is.null(c_known)) {
      stop("pu_method = 'scar' requires c_known (the labeling rate P(S=1|A=1)).")
    }
    pu_fit <- fit_scar_mle(Z_tr, S1, c_known)
    if (anyNA(pu_fit$theta)) stop("SCAR MLE failed.")
    theta  <- pu_fit$theta
    piA    <- as.vector(plogis(cbind(1, X) %*% theta))
    Xi_ev  <- xi_scar_eval(theta, c_known, Z_tr, S1, Z_ev, S_ev)
    return(list(piA_hat = piA, Xi_ev = Xi_ev, Z_ev = Z_ev,
                theta = theta, converged = pu_fit$converged, pu_fit = pu_fit))
  }

  stop("Unknown pu_method: ", pu_method,
       ". Use 'sarem', 'scar', or provide piA_hat directly.")
}

#' One-sample ATE estimation (end-to-end)
#'
#' Full pipeline: sample split, piA estimation, nuisance estimation, ATE with
#' the corrected (Theorem 3) SE.
#'
#' @param X Feature matrix (n x p)
#' @param S PU labels (0/1 vector)
#' @param Y Outcome vector
#' @param pu_method piA estimation method: "sarem" (SAR-EM via Python) or
#'   "scar" (scaled-logistic MLE, pure R, requires c_known)
#' @param piA_hat Optional pre-estimated P(A=1|X) vector of length n.
#'   If provided, piA estimation is skipped and the theta-correction is
#'   omitted; the reported SE is then valid only if piA is known or estimated
#'   at o_p(n^{-1/2}) rate (condition (R1) of Theorem 2).
#' @param c_known Labeling rate c = P(S=1|A=1); required for pu_method = "scar"
#' @param split_idx Optional pre-specified Set 1 indices (default: random n/2)
#' @param max_its SAR-EM max iterations (default: 500)
#' @param C SAR-EM inverse L2 regularization (default: 1.0); also sets the
#'   ridge 1/(n_tr*C) in the penalized information for xi_theta
#' @param trunc Truncation bounds (default: c(0.001, 0.999))
#' @return List with est, se, ci_lower, ci_upper, piA_hat, piS_hat, mu1_hat,
#'         split_idx, theta, converged, pu_fit
#' @export
ate_onesample <- function(X, S, Y, pu_method = "sarem",
                          piA_hat = NULL, c_known = NULL,
                          split_idx = NULL,
                          max_its = 500, C = 1.0,
                          trunc = c(0.001, 0.999)) {
  n <- nrow(X)
  if (is.null(split_idx)) {
    split_idx <- sample(1:n, floor(n / 2))
  }

  X1 <- X[split_idx, , drop = FALSE]
  S1 <- S[split_idx]
  Y1 <- Y[split_idx]

  # --- piA estimation on Set 1, predict on full X ---
  pa <- .estimate_piA(X, S, pu_method, piA_hat, c_known, split_idx, max_its, C)

  # --- Nuisance estimation on Set 1 ---
  df1 <- data.frame(X1)
  df_full <- data.frame(X)

  # piS = P(S=1|X)
  df1$S <- S1
  fit_piS <- glm(S ~ ., data = df1, family = binomial())
  piS_hat <- predict(fit_piS, newdata = df_full, type = "response")

  # mu1 = E[Y|S=1, X]
  labeled <- which(S1 == 1)
  df1_lab <- data.frame(X1[labeled, , drop = FALSE])
  df1_lab$Y <- Y1[labeled]
  fit_mu1 <- lm(Y ~ ., data = df1_lab)
  mu1_hat <- predict(fit_mu1, newdata = df_full)

  # --- ATE estimation on Set 2 (corrected SE) ---
  res <- proposed_estimator_onesample(S, Y, pa$piA_hat, piS_hat,
                                      split_idx, mu1_hat,
                                      Z_ev = pa$Z_ev, Xi_ev = pa$Xi_ev,
                                      trunc = trunc)

  list(
    est       = unname(res["est"]),
    se        = unname(res["se"]),
    ci_lower  = unname(res["ci_lower"]),
    ci_upper  = unname(res["ci_upper"]),
    piA_hat   = pa$piA_hat,
    piS_hat   = piS_hat,
    mu1_hat   = mu1_hat,
    split_idx = split_idx,
    theta     = pa$theta,
    converged = pa$converged,
    pu_fit    = pa$pu_fit
  )
}


#' Two-sample ATE estimation (end-to-end)
#'
#' Full pipeline for case-control design: sample split, piA estimation,
#' nuisance estimation, ATE (tau^UL) with the corrected (Theorem S3) SE.
#'
#' @param X Feature matrix (n x p)
#' @param S PU labels (0/1 vector)
#' @param Y Outcome vector
#' @param pu_method piA estimation method: "sarem" (SAR-EM via Python) or
#'   "scar" (scaled-logistic MLE, pure R, requires c_known)
#' @param piA_hat Optional pre-estimated P(A=1|X) vector of length n.
#'   If provided, piA estimation is skipped and the theta-correction is
#'   omitted; the reported SE is then valid only if piA is known or estimated
#'   at o_p(n^{-1/2}) rate (condition (R1) of Theorem 2).
#' @param c_known Labeling rate c = P(S=1|A=1); required for pu_method = "scar"
#' @param split_idx Optional pre-specified Set 1 indices (default: random n/2)
#' @param max_its SAR-EM max iterations (default: 500)
#' @param C SAR-EM inverse L2 regularization (default: 1.0); also sets the
#'   ridge 1/(n_tr*C) in the penalized information for xi_theta
#' @param trunc Truncation bounds (default: c(0.001, 0.999))
#' @return List with est, se, ci_lower, ci_upper, piA_hat, piS_hat, mu_hat,
#'         mu1_hat, split_idx, theta, converged, pu_fit
#' @export
ate_twosample <- function(X, S, Y, pu_method = "sarem",
                          piA_hat = NULL, c_known = NULL,
                          split_idx = NULL,
                          max_its = 500, C = 1.0,
                          trunc = c(0.001, 0.999)) {
  n <- nrow(X)
  if (is.null(split_idx)) {
    split_idx <- sample(1:n, floor(n / 2))
  }

  X1 <- X[split_idx, , drop = FALSE]
  S1 <- S[split_idx]
  Y1 <- Y[split_idx]

  # --- piA estimation on Set 1, predict on full X ---
  pa <- .estimate_piA(X, S, pu_method, piA_hat, c_known, split_idx, max_its, C)

  # --- Nuisance estimation on Set 1 ---
  df1 <- data.frame(X1)
  df_full <- data.frame(X)

  # piS = P(S=1|X)
  df1$S <- S1
  fit_piS <- glm(S ~ ., data = df1, family = binomial())
  piS_hat <- predict(fit_piS, newdata = df_full, type = "response")

  # mu = E[Y|X] (marginal outcome model, all Set 1)
  df1_mu <- data.frame(X1)
  df1_mu$Y <- Y1
  fit_mu <- lm(Y ~ ., data = df1_mu)
  mu_hat <- predict(fit_mu, newdata = df_full)

  # mu1 = E[Y|S=1, X]
  labeled <- which(S1 == 1)
  df1_lab <- data.frame(X1[labeled, , drop = FALSE])
  df1_lab$Y <- Y1[labeled]
  fit_mu1 <- lm(Y ~ ., data = df1_lab)
  mu1_hat <- predict(fit_mu1, newdata = df_full)

  # --- ATE estimation on Set 2 (corrected SE) ---
  res <- proposed_estimator_twosample(S, Y, pa$piA_hat, piS_hat,
                                      mu_hat, mu1_hat, split_idx,
                                      Z_ev = pa$Z_ev, Xi_ev = pa$Xi_ev,
                                      trunc = trunc)

  list(
    est       = unname(res["est"]),
    se        = unname(res["se"]),
    ci_lower  = unname(res["ci_lower"]),
    ci_upper  = unname(res["ci_upper"]),
    piA_hat   = pa$piA_hat,
    piS_hat   = piS_hat,
    mu_hat    = mu_hat,
    mu1_hat   = mu1_hat,
    split_idx = split_idx,
    theta     = pa$theta,
    converged = pa$converged,
    pu_fit    = pa$pu_fit
  )
}
