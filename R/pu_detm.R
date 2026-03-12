# pu_detm.R
# DETM (Double Exponential Tilting Model) wrapper via PUEM package
#
# PREREQUISITE: PUEM R package must be installed from GitHub.
# See README.md for installation instructions.

#' Fit DETM model via PUEM package
#'
#' @param X Feature matrix (n x p)
#' @param S PU labels (0/1 vector of length n)
#' @return List with model parameters and convergence info
#' @export
fit_detm <- function(X, S) {
  if (!requireNamespace("PUEM", quietly = TRUE)) {
    stop("Package 'PUEM' is required for DETM. See README.md for installation.")
  }

  posdata <- X[S == 1, , drop = FALSE]
  negdata <- X[S == 0, , drop = FALSE]

  if (nrow(posdata) < 2) {
    return(list(converged = FALSE, error = TRUE))
  }

  tryCatch({
    p <- ncol(posdata)
    res <- PUEM::PUfit(
      posdata = as.matrix(posdata),
      negdata = as.matrix(negdata),
      pival   = 0.5,
      alp1    = 0, beta1 = rep(0, p),
      alp2    = 0, beta2 = rep(0, p)
    )
    list(
      pival  = res$out_detm_full$pival,
      alp1   = res$out_detm_full$alp1, beta1 = res$out_detm_full$beta1,
      alp2   = res$out_detm_full$alp2, beta2 = res$out_detm_full$beta2,
      n_pos  = nrow(posdata),
      m_neg  = nrow(negdata),
      converged = TRUE,
      error     = FALSE
    )
  }, error = function(e) {
    warning(paste("DETM failed:", e$message))
    list(converged = FALSE, error = TRUE)
  })
}

#' Predict P(A=1|X) from DETM model
#'
#' @param detm_fit Result from fit_detm()
#' @param X_new New feature matrix
#' @return Vector of estimated P(A=1|X)
#' @export
predict_detm <- function(detm_fit, X_new) {
  if (detm_fit$error) stop("Cannot predict: DETM fit failed.")

  X_new <- as.matrix(X_new)
  Lambda1 <- exp(detm_fit$alp1 + X_new %*% detm_fit$beta1)
  Lambda2 <- exp(detm_fit$alp2 + X_new %*% detm_fit$beta2)
  n_pos <- detm_fit$n_pos
  m_neg <- detm_fit$m_neg
  pi_est <- detm_fit$pival

  base_ratio <- n_pos / (n_pos + m_neg)
  neg_ratio  <- m_neg / (n_pos + m_neg)

  denom <- base_ratio + neg_ratio * (pi_est * Lambda1 + (1 - pi_est) * Lambda2)
  prob_pred <- (base_ratio + neg_ratio * pi_est * Lambda1) / denom
  as.vector(prob_pred)
}

#' Predict P(S=1|A=1,X) from DETM model
#'
#' @param detm_fit Result from fit_detm()
#' @param X_new New feature matrix
#' @return Vector of estimated propensity scores e(X)
#' @export
predict_e_detm <- function(detm_fit, X_new) {
  if (detm_fit$error) stop("Cannot predict: DETM fit failed.")

  # e(X) = gamma / (gamma + (1-gamma)*lambda1(X))
  # gamma = P(S=1|A=1) estimated as n_pos / (n_pos + m_neg * pival)
  gamma_hat <- detm_fit$n_pos / (detm_fit$n_pos + detm_fit$m_neg * detm_fit$pival)
  lambda1 <- exp(detm_fit$alp1 + as.matrix(X_new) %*% detm_fit$beta1)
  e_pred <- gamma_hat / (gamma_hat + (1 - gamma_hat) * lambda1)
  as.vector(e_pred)
}
