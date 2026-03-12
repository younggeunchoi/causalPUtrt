# pu_models.R
# PU learning models using regularized IRWLS

#' Fit logistic regression for PU learning using regularized IRWLS
#'
#' @param x Feature matrix (n x p)
#' @param s PU labels (0/1 vector)
#' @param e Propensity scores (optional)
#' @param sample_weight Sample weights (optional)
#' @param C Inverse regularization strength (default: 1.0)
#' @param ... Additional arguments for IRWLS
#' @return Fitted model
fit_logistic_pu <- function(x, s, e = NULL, sample_weight = NULL,
                            C = 1.0, ...) {
  x <- as.matrix(x)
  s <- as.vector(s)

  if (is.null(e)) {
    model <- logistic_regression_irwls_reg(x, s, weights = sample_weight,
                                           C = C, ...)
  } else {
    e <- as.vector(e)
    n <- length(s)

    weights_pos <- ifelse(s == 1, 1 / e, 0)
    weights_neg <- ifelse(s == 1, 1 - 1 / e, 1)

    if (!is.null(sample_weight)) {
      sample_weight <- as.vector(sample_weight)
      weights_pos <- sample_weight * weights_pos
      weights_neg <- sample_weight * weights_neg
    }

    x_doubled <- rbind(x, x)
    y_doubled <- c(rep(1, n), rep(0, n))
    weights_all <- c(weights_pos, weights_neg)

    model <- logistic_regression_irwls_reg(x_doubled, y_doubled,
                                           weights = weights_all,
                                           C = C, ...)
  }

  class(model) <- c("logistic_pu_irwls_reg", class(model))
  return(model)
}

#' Predict probabilities for PU model
#'
#' @param model Fitted model
#' @param x New data
#' @return Predicted probabilities
predict_proba_pu <- function(model, x) {
  return(predict_proba_irwls_reg(model, x))
}
