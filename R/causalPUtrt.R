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

# PU learning methods
source(file.path(.causalPUtrt_dir, "pu_sarem.R"))    # Requires reticulate + Python sarpu

# ATE estimators
source(file.path(.causalPUtrt_dir, "ate_estimators.R"))
source(file.path(.causalPUtrt_dir, "ate_wrappers.R"))

message("causalPUtrt loaded. PU method: SAR-EM (Python). Use piA_hat param for custom estimates.")
message("  - SAR-EM: call init_sarem(python_env, sarpu_path) before use")
