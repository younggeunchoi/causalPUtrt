# ate_wrappers.R
# End-to-end wrapper functions for ATE estimation with PU learning.
#
# These wrappers handle the full pipeline:
#   sample splitting -> PU fitting -> nuisance estimation -> ATE calculation

#' One-sample ATE estimation (end-to-end)
#'
#' Full pipeline: sample split, PU learning, nuisance estimation, ATE.
#'
#' @param X Feature matrix (n x p)
#' @param S PU labels (0/1 vector)
#' @param Y Outcome vector
#' @param pu_method PU learning method: "tm", "sarem", or "detm"
#' @param split_idx Optional pre-specified Set 1 indices (default: random n/2)
#' @param epochs TM epochs (default: 500)
#' @param max_its SAR-EM max iterations (default: 500)
#' @param C SAR-EM regularization (default: 1.0)
#' @param trunc Truncation bounds (default: c(0.001, 0.999))
#' @return List with est, se, ci_lower, ci_upper, piA_hat, piS_hat, mu1_hat,
#'         split_idx, pu_fit
#' @export
ate_onesample <- function(X, S, Y, pu_method = "tm",
                          split_idx = NULL,
                          epochs = 500, max_its = 500, C = 1.0,
                          trunc = c(0.001, 0.999)) {
  n <- nrow(X)
  if (is.null(split_idx)) {
    split_idx <- sample(1:n, floor(n / 2))
  }

  X1 <- X[split_idx, , drop = FALSE]
  S1 <- S[split_idx]
  Y1 <- Y[split_idx]

  # --- PU learning on Set 1, predict on full X ---
  pu_fit <- NULL
  if (pu_method == "tm") {
    pu_fit  <- pu_learn_tm(X1, S1, epochs = epochs)
    piA_hat <- predict_proba_pu(pu_fit$model_clf, X)
  } else if (pu_method == "sarem") {
    pu_fit  <- fit_sarem(X1, S1, max_its = max_its, C = C)
    if (pu_fit$error) stop("SAR-EM failed to converge.")
    preds   <- predict_sarem(pu_fit, X)
    piA_hat <- preds$prob_pred
  } else if (pu_method == "detm") {
    pu_fit  <- fit_detm(X1, S1)
    if (pu_fit$error) stop("DETM failed to converge.")
    piA_hat <- predict_detm(pu_fit, X)
  } else {
    stop("Unknown pu_method: ", pu_method, ". Use 'tm', 'sarem', or 'detm'.")
  }

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

  # --- ATE estimation on Set 2 ---
  res <- proposed_estimator_onesample(S, Y, piA_hat, piS_hat,
                                      split_idx, mu1_hat, trunc = trunc)

  list(
    est      = unname(res["est"]),
    se       = unname(res["se"]),
    ci_lower = unname(res["ci_lower"]),
    ci_upper = unname(res["ci_upper"]),
    piA_hat  = piA_hat,
    piS_hat  = piS_hat,
    mu1_hat  = mu1_hat,
    split_idx = split_idx,
    pu_fit   = pu_fit
  )
}


#' Two-sample ATE estimation (end-to-end)
#'
#' Full pipeline for case-control design: sample split, PU learning,
#' nuisance estimation, ATE.
#'
#' @param X Feature matrix (n x p)
#' @param S PU labels (0/1 vector)
#' @param Y Outcome vector
#' @param pu_method PU learning method: "tm", "sarem", or "detm"
#' @param split_idx Optional pre-specified Set 1 indices (default: random n/2)
#' @param epochs TM epochs (default: 500)
#' @param max_its SAR-EM max iterations (default: 500)
#' @param C SAR-EM regularization (default: 1.0)
#' @param trunc Truncation bounds (default: c(0.001, 0.999))
#' @return List with est, se, ci_lower, ci_upper, piA_hat, piS_hat, mu_hat,
#'         mu1_hat, split_idx, pu_fit
#' @export
ate_twosample <- function(X, S, Y, pu_method = "tm",
                          split_idx = NULL,
                          epochs = 500, max_its = 500, C = 1.0,
                          trunc = c(0.001, 0.999)) {
  n <- nrow(X)
  if (is.null(split_idx)) {
    split_idx <- sample(1:n, floor(n / 2))
  }

  X1 <- X[split_idx, , drop = FALSE]
  S1 <- S[split_idx]
  Y1 <- Y[split_idx]

  # --- PU learning on Set 1, predict on full X ---
  pu_fit <- NULL
  if (pu_method == "tm") {
    pu_fit  <- pu_learn_tm(X1, S1, epochs = epochs)
    piA_hat <- predict_proba_pu(pu_fit$model_clf, X)
  } else if (pu_method == "sarem") {
    pu_fit  <- fit_sarem(X1, S1, max_its = max_its, C = C)
    if (pu_fit$error) stop("SAR-EM failed to converge.")
    preds   <- predict_sarem(pu_fit, X)
    piA_hat <- preds$prob_pred
  } else if (pu_method == "detm") {
    pu_fit  <- fit_detm(X1, S1)
    if (pu_fit$error) stop("DETM failed to converge.")
    piA_hat <- predict_detm(pu_fit, X)
  } else {
    stop("Unknown pu_method: ", pu_method, ". Use 'tm', 'sarem', or 'detm'.")
  }

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

  # --- ATE estimation on Set 2 ---
  res <- proposed_estimator_twosample(S, Y, piA_hat, piS_hat,
                                      mu_hat, mu1_hat,
                                      split_idx, trunc = trunc)

  list(
    est      = unname(res["est"]),
    se       = unname(res["se"]),
    ci_lower = unname(res["ci_lower"]),
    ci_upper = unname(res["ci_upper"]),
    piA_hat  = piA_hat,
    piS_hat  = piS_hat,
    mu_hat   = mu_hat,
    mu1_hat  = mu1_hat,
    split_idx = split_idx,
    pu_fit   = pu_fit
  )
}
