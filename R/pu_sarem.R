# pu_sarem.R
# SAR-EM wrapper via reticulate (Python sarpu package)
#
# PREREQUISITE: Python sarpu package must be installed.
# See README.md for installation instructions.

#' Initialize SAR-EM Python environment
#'
#' Must be called once before using fit_sarem() or predict_sarem().
#'
#' @param virtualenv_path Path to Python virtualenv where sarpu is installed.
#'   e.g., "/path/to/your/venv"
#' @param sarpu_path Path to sarpu source directory (containing sarpu/ package).
#'   e.g., "/path/to/SAR_PU_python/sarpu"
#' @export
init_sarem <- function(virtualenv_path, sarpu_path) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required for SAR-EM. Install with: install.packages('reticulate')")
  }

  reticulate::use_virtualenv(virtualenv_path, required = TRUE)

  reticulate::py_run_string(sprintf('
import sys
sys.path.insert(0, "%s")
import numpy as np
from sarpu.pu_learning import pu_learn_sar_em
from sarpu.PUmodels import LogisticRegressionPU

def _run_sarem(X, S, max_its=500, C=1.0):
    X = np.array(X)
    S = np.array(S, dtype=int)

    class_model = LogisticRegressionPU(C=C, max_iter=500)
    prop_model  = LogisticRegressionPU(C=C, max_iter=500)

    propensity_attributes     = np.ones(X.shape[1], dtype=bool)
    classification_attributes = np.ones(X.shape[1], dtype=bool)

    classification_result, propensity_result, info = pu_learn_sar_em(
        X, S,
        propensity_attributes=propensity_attributes,
        classification_attributes=classification_attributes,
        classification_model=class_model,
        propensity_model=prop_model,
        max_its=max_its,
        convergence_window=10,
        refit_classifier=True
    )

    prob_pred = classification_result.predict_proba(X)
    if prob_pred.ndim > 1:
        prob_pred = prob_pred[:, 1]

    prop_pred = propensity_result.predict_proba(X)
    if prop_pred.ndim > 1:
        prop_pred = prop_pred[:, 1]

    return {
        "prob_pred": prob_pred,
        "prop_pred": prop_pred,
        "converged": info.get("converged", info["nb_iterations"] < max_its - 1),
        "iterations": info["nb_iterations"],
        "class_model": classification_result,
        "prop_model": propensity_result
    }

def _predict_sarem(result, X_new):
    X_new = np.array(X_new)

    prob_pred = result["class_model"].predict_proba(X_new)
    if prob_pred.ndim > 1:
        prob_pred = prob_pred[:, 1]

    prop_pred = result["prop_model"].predict_proba(X_new)
    if prop_pred.ndim > 1:
        prop_pred = prop_pred[:, 1]

    return {
        "prob_pred": prob_pred,
        "prop_pred": prop_pred
    }
', sarpu_path))

  message("SAR-EM Python environment initialized successfully.")
  invisible(TRUE)
}

#' Fit SAR-EM model via Python
#'
#' Requires init_sarem() to have been called first.
#'
#' @param X Feature matrix (n x p)
#' @param S PU labels (0/1 vector of length n)
#' @param max_its Maximum EM iterations (default: 500)
#' @param C Regularization parameter (default: 1.0)
#' @return List with:
#'   \item{prob_pred}{Estimated P(A=1|X) on training data}
#'   \item{prop_pred}{Estimated P(S=1|A=1,X) on training data}
#'   \item{converged}{Logical, whether EM converged}
#'   \item{iterations}{Number of EM iterations}
#'   \item{py_result}{Raw Python result object (for predict_sarem)}
#'   \item{error}{Logical, whether an error occurred}
#' @export
fit_sarem <- function(X, S, max_its = 500, C = 1.0) {
  tryCatch({
    result <- reticulate::py$`_run_sarem`(X, S,
                                           max_its = as.integer(max_its),
                                           C = C)
    list(
      prob_pred  = as.vector(result$prob_pred),
      prop_pred  = as.vector(result$prop_pred),
      converged  = result$converged,
      iterations = result$iterations,
      py_result  = result,
      error      = FALSE
    )
  }, error = function(e) {
    warning(paste("SAR-EM failed:", e$message))
    list(
      prob_pred  = rep(NA, nrow(as.matrix(X))),
      prop_pred  = rep(NA, nrow(as.matrix(X))),
      converged  = FALSE,
      iterations = NA,
      py_result  = NULL,
      error      = TRUE
    )
  })
}

#' Predict on new data using a fitted SAR-EM model
#'
#' @param sarem_fit Result from fit_sarem()
#' @param X_new New feature matrix
#' @return List with prob_pred (piA) and prop_pred (e) on new data
#' @export
predict_sarem <- function(sarem_fit, X_new) {
  if (sarem_fit$error || is.null(sarem_fit$py_result)) {
    stop("Cannot predict: SAR-EM fit failed or py_result is NULL.")
  }
  result <- reticulate::py$`_predict_sarem`(sarem_fit$py_result, X_new)
  list(
    prob_pred = as.vector(result$prob_pred),
    prop_pred = as.vector(result$prop_pred)
  )
}
