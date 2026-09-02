# pu_sarem.R
# SAR-EM wrapper via reticulate (Python sarpu package).
#
# The EM is run WITHOUT the final classifier refit (refit_classifier=False):
# the refit step uses negative weights (w_i = (1-S_i) + S_i(1-1/e_i), negative
# whenever e_i > 1), which makes the level of piA_hat inconsistent. At the EM
# fixed point, (theta, phi) solves the L2-penalized joint estimating equation,
# so its influence function is available for the corrected variance
# (see xi_sar_eval below and README).
#
# PREREQUISITE: Python sarpu package must be installed.
# See README.md for installation instructions.

#' Initialize SAR-EM Python environment
#'
#' Must be called once before using fit_sarem().
#'
#' @param python_env Path to Python virtualenv where sarpu is installed.
#'   e.g., "/path/to/your/venv"
#' @param sarpu_path Path to sarpu source directory (containing sarpu/ package).
#'   e.g., "/path/to/SAR-PU/sarpu"
#' @export
init_sarem <- function(python_env, sarpu_path) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required for SAR-EM. Install with: install.packages('reticulate')")
  }

  reticulate::use_virtualenv(python_env, required = TRUE)

  reticulate::py_run_string(sprintf('
import sys
sys.path.insert(0, "%s")
import numpy as np
from sarpu.PUmodels import LogisticRegressionPU
from sarpu.pu_learning import pu_learn_sar_em

def _run_sarem_norefit(X, S, class_attr, prop_attr, C=1.0, max_its=500):
    """SAR-EM without the classifier refit (EM fixed point). Returns class/prop
    coefficients (on the selected attributes) and the convergence flag."""
    X = np.array(X); S = np.array(S, dtype=int)
    ca = np.array(class_attr, dtype=bool); pa = np.array(prop_attr, dtype=bool)
    cm = LogisticRegressionPU(C=C, max_iter=500); pm = LogisticRegressionPU(C=C, max_iter=500)
    cr, pr, info = pu_learn_sar_em(X, S, propensity_attributes=pa,
        classification_attributes=ca, classification_model=cm, propensity_model=pm,
        max_its=max_its, convergence_window=10, refit_classifier=False)
    m = cr.model if hasattr(cr, "model") else cr
    pmod = pr.model if hasattr(pr, "model") else pr
    return {"class_coef": [float(m.intercept_[0])] + [float(v) for v in m.coef_.flatten()],
            "prop_coef":  [float(pmod.intercept_[0])] + [float(v) for v in pmod.coef_.flatten()],
            "converged": bool(info.get("converged", info["nb_iterations"] < max_its - 1))}
', sarpu_path))

  message("SAR-EM Python environment initialized successfully.")
  invisible(TRUE)
}

#' Fit SAR-EM model via Python (no classifier refit)
#'
#' Requires init_sarem() to have been called first. Only the logistic
#' coefficients cross the R/Python boundary; predictions are computed in R
#' via plogis().
#'
#' @param X Feature matrix (n x p)
#' @param S PU labels (0/1 vector of length n)
#' @param max_its Maximum EM iterations (default: 500)
#' @param C Inverse L2 regularization strength (default: 1.0)
#' @param class_attr Logical vector (length p): features used by the
#'   classification model P(A=1|X). Default NULL = all features.
#' @param prop_attr Logical vector (length p): features used by the
#'   propensity model P(S=1|A=1,X). Default NULL = all features.
#' @return List with:
#'   \item{theta}{Classification coefficients (intercept, selected features)}
#'   \item{phi}{Propensity coefficients (intercept, selected features)}
#'   \item{class_attr, prop_attr}{The feature masks used}
#'   \item{converged}{Logical, whether EM converged}
#'   \item{error}{Logical, whether an error occurred}
#' @export
fit_sarem <- function(X, S, max_its = 500, C = 1.0,
                      class_attr = NULL, prop_attr = NULL) {
  X <- as.matrix(X)
  if (is.null(class_attr)) class_attr <- rep(TRUE, ncol(X))
  if (is.null(prop_attr))  prop_attr  <- rep(TRUE, ncol(X))
  tryCatch({
    res <- reticulate::py$`_run_sarem_norefit`(X, S, class_attr, prop_attr,
                                               C = C,
                                               max_its = as.integer(max_its))
    list(
      theta      = as.numeric(res$class_coef),
      phi        = as.numeric(res$prop_coef),
      class_attr = class_attr,
      prop_attr  = prop_attr,
      converged  = isTRUE(res$converged),
      error      = FALSE
    )
  }, error = function(e) {
    warning(paste("SAR-EM failed:", e$message))
    list(
      theta      = rep(NA_real_, sum(class_attr) + 1),
      phi        = rep(NA_real_, sum(prop_attr) + 1),
      class_attr = class_attr,
      prop_attr  = prop_attr,
      converged  = FALSE,
      error      = TRUE
    )
  })
}

#' Predict on new data using a fitted SAR-EM model
#'
#' Pure R: probabilities are reconstructed from the fitted coefficients.
#'
#' @param sarem_fit Result from fit_sarem()
#' @param X_new New feature matrix
#' @return List with prob_pred (piA) and prop_pred (e) on new data
#' @export
predict_sarem <- function(sarem_fit, X_new) {
  if (sarem_fit$error) stop("Cannot predict: SAR-EM fit failed.")
  X_new <- as.matrix(X_new)
  Zc <- cbind(1, X_new[, sarem_fit$class_attr, drop = FALSE])
  Wc <- cbind(1, X_new[, sarem_fit$prop_attr,  drop = FALSE])
  list(
    prob_pred = as.vector(plogis(Zc %*% sarem_fit$theta)),
    prop_pred = as.vector(plogis(Wc %*% sarem_fit$phi))
  )
}

#' theta-block influence function values for SAR-EM (norefit)
#'
#' METHOD-DEPENDENT: this xi_theta is derived for the SAR-EM (norefit)
#' estimator, whose (theta, phi) solves the L2-penalized joint logistic
#' estimating equation. It is NOT valid for other piA estimation procedures
#' (see README).
#'
#' The penalized joint information (theta and phi blocks, ridge = 1/(n_tr*C)
#' matching sklearn's L2 penalty) is assembled from the TRAINING half; the
#' joint score is evaluated at the EVAL observations. Returns the theta block.
#'
#' @param theta Classification coefficients (intercept first)
#' @param phi Propensity coefficients (intercept first)
#' @param Z_tr,W_tr Design matrices (with intercept column) on the training half
#' @param S_tr PU labels on the training half
#' @param Z_ev,W_ev Design matrices (with intercept column) on the eval half
#' @param S_ev PU labels on the eval half
#' @param ridge Ridge added to the information blocks; use 1/(n_tr*C)
#' @return n_ev x length(theta) matrix of xi_theta values
#' @export
xi_sar_eval <- function(theta, phi, Z_tr, W_tr, S_tr, Z_ev, W_ev, S_ev, ridge) {
  n_tr <- nrow(Z_tr); pt <- ncol(Z_tr); pp <- ncol(W_tr)
  f_tr <- clip(plogis(as.vector(Z_tr %*% theta)))
  e_tr <- clip(plogis(as.vector(W_tr %*% phi)))
  q <- f_tr * e_tr; denom <- 1 - q; omega <- q / denom
  I_tt <- crossprod(Z_tr, Z_tr * (omega * (1 - f_tr)^2)) / n_tr + ridge * diag(pt)
  I_tp <- crossprod(Z_tr, W_tr * (omega * (1 - f_tr) * (1 - e_tr))) / n_tr
  I_pp <- crossprod(W_tr, W_tr * (omega * (1 - e_tr)^2)) / n_tr + ridge * diag(pp)
  I_full <- rbind(cbind(I_tt, I_tp), cbind(t(I_tp), I_pp))
  f_ev <- clip(plogis(as.vector(Z_ev %*% theta)))
  e_ev <- clip(plogis(as.vector(W_ev %*% phi)))
  q_ev <- f_ev * e_ev; r_ev <- (S_ev - q_ev) / (1 - q_ev)
  U_ev <- cbind(Z_ev * (r_ev * (1 - f_ev)), W_ev * (r_ev * (1 - e_ev)))
  Xi_full <- t(solve(I_full, t(U_ev)))
  Xi_full[, seq_len(pt), drop = FALSE]
}
