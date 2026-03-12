# pu_utils.R
# Utility functions for PU learning

#' Select features from matrix based on boolean vector
#'
#' @param x Feature matrix
#' @param features Boolean vector indicating which features to select
#' @return Subset of x with selected features
select_features <- function(x, features) {
  if (is.null(features)) return(x)
  if (length(features) == 0 || sum(features) == 0) {
    return(matrix(nrow = nrow(x), ncol = 0))
  }
  x[, features, drop = FALSE]
}

#' Make propensity weighted data for PU learning
#'
#' @param x Feature matrix (n x p)
#' @param s PU labels (0/1 vector of length n)
#' @param e Propensity scores (vector of length n)
#' @param sample_weight Optional sample weights
#' @return List with Xp (expanded features), Yp (expanded labels), Wp (weights)
make_propensity_weighted_data <- function(x, s, e, sample_weight = NULL) {
  n <- length(s)

  weights_pos <- ifelse(s == 1, 1 / e, 0)
  weights_neg <- ifelse(s == 1, 1 - 1 / e, 1)

  if (!is.null(sample_weight)) {
    weights_pos <- sample_weight * weights_pos
    weights_neg <- sample_weight * weights_neg
  }

  Xp <- rbind(x, x)
  Yp <- c(rep(1, n), rep(0, n))
  Wp <- c(weights_pos, weights_neg)

  keep_idx <- which(Wp > 0)

  return(list(
    Xp = Xp[keep_idx, , drop = FALSE],
    Yp = Yp[keep_idx],
    Wp = Wp[keep_idx]
  ))
}

#' Logistic (sigmoid) function
sigmoid <- function(x) {
  1 / (1 + exp(-x))
}
