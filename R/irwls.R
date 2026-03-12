# irwls.R
# IRWLS with L2 regularization matching scikit-learn's formulation

#' Logistic regression using IRWLS with L2 regularization
#'
#' Minimizes: (1/S) * NLL + (1/(2*S*C)) * ||beta||^2
#' where S = sum(weights) and NLL is negative log-likelihood
#'
#' @param X Feature matrix (n x p)
#' @param y Binary response (0/1)
#' @param weights Sample weights (default: all 1)
#' @param C Inverse regularization strength (default: 1.0)
#' @param max_iter Maximum iterations (default: 100)
#' @param tol Convergence tolerance (default: 1e-6)
#' @param fit_intercept Whether to fit intercept (default: TRUE)
#' @return List with coef, intercept, iterations, converged
logistic_regression_irwls_reg <- function(X, y, weights = NULL, C = 1.0,
                                          max_iter = 100, tol = 1e-6,
                                          fit_intercept = TRUE) {
  X <- as.matrix(X)
  y <- as.vector(y)
  n <- nrow(X)
  p <- ncol(X)

  if (is.null(weights)) {
    weights <- rep(1, n)
  } else {
    weights <- as.vector(weights)
  }

  S <- sum(abs(weights))

  if (fit_intercept) {
    X <- cbind(1, X)
    p <- p + 1
  }

  beta <- rep(0, p)

  if (fit_intercept) {
    lambda_vec <- c(0, rep(1 / (S * C), p - 1))
  } else {
    lambda_vec <- rep(1 / (S * C), p)
  }
  if (length(lambda_vec) == 1) {
    Lambda <- matrix(lambda_vec, 1, 1)
  } else {
    Lambda <- diag(lambda_vec)
  }

  converged <- FALSE
  for (iter in 1:max_iter) {
    eta <- X %*% beta
    mu <- plogis(eta)
    mu <- pmax(mu, 1e-15)
    mu <- pmin(mu, 1 - 1e-15)

    W <- as.vector(mu * (1 - mu))
    W_combined <- weights * W / S
    z <- eta + (y - mu) / W

    XtW <- t(X * W_combined)
    XtWX <- XtW %*% X
    XtWz <- XtW %*% z

    XtWX_reg <- XtWX + Lambda
    beta_new <- solve(XtWX_reg, XtWz)

    if (max(abs(beta_new - beta)) < tol) {
      converged <- TRUE
      break
    }
    beta <- beta_new
  }
  if (iter == max_iter) converged <- FALSE

  if (fit_intercept) {
    intercept <- beta[1]
    coef <- beta[-1]
  } else {
    intercept <- 0
    coef <- beta
  }

  return(list(
    coef = coef,
    intercept = intercept,
    iterations = iter,
    converged = converged,
    C = C
  ))
}

#' Predict probabilities from IRWLS model
#'
#' @param model Fitted model from logistic_regression_irwls_reg
#' @param X New data matrix
#' @return Predicted probabilities
predict_proba_irwls_reg <- function(model, X) {
  X <- as.matrix(X)
  eta <- model$intercept + X %*% model$coef
  return(plogis(eta))
}
