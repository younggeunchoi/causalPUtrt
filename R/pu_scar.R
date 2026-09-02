# pu_scar.R
# SCAR (Selected Completely At Random) treatment-propensity estimation.
#
# Model: piS(X; theta) = c * expit(theta' Z) with the labeling rate
# c = P(S=1|A=1) KNOWN (supplied by the user), which induces the parametric
# treatment propensity piA(X; theta) = expit(theta' Z).
#
# theta is estimated by maximum likelihood of S ~ Bernoulli(c * expit(theta'Z))
# (BFGS with analytic gradient, warm-started at glm coefficients). Its MLE
# influence function (xi_scar_eval) feeds the Theorem-3 corrected variance.
# Pure R, no external dependency.

#' Fit the SCAR scaled-logistic MLE
#'
#' Maximizes the likelihood of S ~ Bernoulli(c * expit(theta' Z)) on the
#' training half. The labeling rate c = P(S=1|A=1) must be known.
#'
#' @param Zc Design matrix INCLUDING the intercept column (n_tr x p)
#' @param S PU labels (0/1 vector) on the training half
#' @param c_known Known labeling rate c = P(S=1|A=1)
#' @return List with theta (coefficients) and converged (logical)
#' @export
fit_scar_mle <- function(Zc, S, c_known) {
  negll <- function(th) {
    m <- clip(c_known * plogis(as.vector(Zc %*% th)), 1e-10)
    -mean(S * log(m) + (1 - S) * log(1 - m))
  }
  grad <- function(th) {
    s <- plogis(as.vector(Zc %*% th))
    cs <- clip(c_known * s, 1e-10)
    -colMeans(Zc * ((1 - s) * (S - (1 - S) * cs / (1 - cs))))
  }
  # start at glm coefs (fast, robust)
  st <- tryCatch(unname(coef(glm.fit(Zc, S, family = binomial()))),
                 error = function(e) rep(0, ncol(Zc)))
  st[!is.finite(st)] <- 0
  o <- tryCatch(optim(st, negll, grad, method = "BFGS", control = list(maxit = 300)),
                error = function(e) NULL)
  if (is.null(o)) return(list(theta = rep(NA_real_, ncol(Zc)), converged = FALSE))
  list(theta = o$par, converged = (o$convergence == 0))
}

#' Influence function values for the SCAR scaled-logistic MLE
#'
#' METHOD-DEPENDENT: this xi_theta is the MLE influence function of the SCAR
#' scaled-logistic model with known c. It is NOT valid for other piA
#' estimation procedures (see README).
#'
#'   score u_i = (1-sigma_i) { S_i - (1-S_i) c sigma_i / (1 - c sigma_i) } Z_i
#'   info  I   = mean_train[ c sigma (1-sigma)^2 / (1 - c sigma) Z Z' ]
#'
#' The information is computed on the TRAINING half; the score is evaluated
#' at the EVAL observations.
#'
#' @param theta Fitted coefficients from fit_scar_mle()
#' @param c_known Known labeling rate c = P(S=1|A=1)
#' @param Z_tr Design matrix (with intercept column) on the training half
#' @param S_tr PU labels on the training half
#' @param Z_ev Design matrix (with intercept column) on the eval half
#' @param S_ev PU labels on the eval half
#' @return n_ev x length(theta) matrix of xi_theta values
#' @export
xi_scar_eval <- function(theta, c_known, Z_tr, S_tr, Z_ev, S_ev) {
  s_tr <- clip(plogis(as.vector(Z_tr %*% theta)))
  w <- c_known * s_tr * (1 - s_tr)^2 / clip(1 - c_known * s_tr, 1e-10)
  I <- crossprod(Z_tr, Z_tr * w) / nrow(Z_tr)
  s_ev <- clip(plogis(as.vector(Z_ev %*% theta)))
  cs <- clip(c_known * s_ev, 1e-10)
  U_ev <- Z_ev * ((1 - s_ev) * (S_ev - (1 - S_ev) * cs / (1 - cs)))
  t(solve(I, t(U_ev)))
}
