# causalPUtrt.R
# Main entry point. Source this file to load all functions.
#
# Usage:
#   source("R/causalPUtrt.R")

# Determine the directory where this script lives
.causalPUtrt_dir <- if (exists(".causalPUtrt_root")) {
  .causalPUtrt_root
} else {
  # Try to infer from source() call
  tryCatch({
    dirname(sys.frame(1)$ofile)
  }, error = function(e) {
    "R"  # fallback: assume working directory is repo root
  })
}

# Core utilities (no external dependencies)
source(file.path(.causalPUtrt_dir, "pu_utils.R"))

# Treatment-propensity (piA) estimation methods
source(file.path(.causalPUtrt_dir, "pu_sarem.R"))    # SAR-EM: requires reticulate + Python sarpu
source(file.path(.causalPUtrt_dir, "pu_scar.R"))     # SCAR:   pure R, requires known c

# ATE estimators
source(file.path(.causalPUtrt_dir, "ate_estimators.R"))
source(file.path(.causalPUtrt_dir, "ate_wrappers.R"))

message("causalPUtrt loaded. piA methods: SAR-EM (Python, norefit), SCAR (pure R, known c).")
message("  - SAR-EM: call init_sarem(python_env, sarpu_path) before use")
message("  - SCAR:   pu_method = 'scar' with c_known = P(S=1|A=1)")
message("  - SE: Theorem-3/S3 corrected (psi + B'xi_theta); xi_theta is method-dependent (see README)")
