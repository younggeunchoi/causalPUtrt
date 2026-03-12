# pu_tm.R
# TM (Two Models) method for biased PU learning
# Pure R implementation, no Python dependency.

#' Prepare weighted PU data for TM method
#'
#' @param x Feature matrix
#' @param s PU labels
#' @param ex Propensity estimates P(S=1|Y=1,X)
#' @param sx Joint probability estimates P(S=1,Y=1|X)
#' @return List with x_temp, s_temp, weights
prepare_weighted_pu_data_tm <- function(x, s, ex, sx) {
  idx_0 <- which(s == 0)
  idx_1 <- which(s == 1)
  n0 <- length(idx_0)
  n1 <- length(idx_1)

  if (n1 == 0 || n0 == 0) {
    stop("Need both labeled and unlabeled examples")
  }

  x0 <- x[idx_0, , drop = FALSE]
  x1 <- x[idx_1, , drop = FALSE]

  s_temp <- c(rep(1, n1), rep(1, n0), rep(0, n0))
  x_temp <- rbind(x1, x0, x0)

  weights1 <- rep(1, n1)
  weights2 <- ((1 - ex[idx_0]) / ex[idx_0]) * (sx[idx_0] / (1 - sx[idx_0]))
  weights3 <- 1 - weights2

  weights2 <- pmax(weights2, 0)
  weights3 <- pmax(weights3, 0)

  weights <- c(weights1, weights2, weights3)
  return(list(x = x_temp, s = s_temp, weights = weights))
}

#' TM (Two Models) method for PU learning
#'
#' Estimates P(A=1|X) and P(S=1|A=1,X) from PU data.
#'
#' @param x Feature matrix (n x p)
#' @param s PU labels (0/1 vector of length n)
#' @param epochs Number of iterations (default 500)
#' @param features_clf Features to use for classification (NULL = all)
#' @param features_ex Features to use for propensity (NULL = all)
#' @return List with model_clf, model_ex, predictions (piA_hat), propensities (e_hat)
pu_learn_tm <- function(x, s, epochs = 500,
                        features_clf = NULL, features_ex = NULL) {

  n <- nrow(x)
  x_clf <- if (!is.null(features_clf)) select_features(x, features_clf) else x
  x_ex  <- if (!is.null(features_ex))  select_features(x, features_ex)  else x

  # Initialize with naive model
  model_naive <- fit_logistic_pu(x_clf, s)
  sx <- predict_proba_pu(model_naive, x_clf)
  ex <- (sx + 1) / 2

  model_clf <- NULL
  model_ex  <- NULL

  for (i in 1:epochs) {
    # Classification model: P(A=1|X)
    weighted_data <- prepare_weighted_pu_data_tm(x_clf, s, ex, sx)
    model_clf <- fit_logistic_pu(weighted_data$x, weighted_data$s,
                                 sample_weight = weighted_data$weights)
    yx <- predict_proba_pu(model_clf, x_clf)

    # Threshold for propensity model
    hat_c <- mean(s)
    yx1 <- yx[s == 1]
    if (length(yx1) > 0 && hat_c > 0 && hat_c < 1) {
      val_thrs <- quantile(yx1, probs = hat_c)
    } else {
      val_thrs <- 0.5
    }

    sel <- union(which(yx > val_thrs), which(s == 1))
    x_sel <- x_ex[sel, , drop = FALSE]
    s_sel <- s[sel]

    # Propensity model: P(S=1|A=1,X)
    model_ex <- fit_logistic_pu(x_sel, s_sel)
    ex <- predict_proba_pu(model_ex, x_ex)

    # Constraint: e(x) >= P(S=1|X)
    too_small <- which(ex < sx)
    if (length(too_small) > 0) ex[too_small] <- sx[too_small]

    sx <- ex * yx
  }

  # Final classification model
  weighted_data <- prepare_weighted_pu_data_tm(x_clf, s, ex, sx)
  model_clf <- fit_logistic_pu(weighted_data$x, weighted_data$s,
                                sample_weight = weighted_data$weights)

  return(list(
    model_clf   = model_clf,
    model_ex    = model_ex,
    predictions  = predict_proba_pu(model_clf, x_clf),
    propensities = predict_proba_pu(model_ex, x_ex)
  ))
}
