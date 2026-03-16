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
source(file.path(.causalPUtrt_dir, "irwls.R"))
source(file.path(.causalPUtrt_dir, "pu_utils.R"))
source(file.path(.causalPUtrt_dir, "pu_models.R"))

# PU learning methods
source(file.path(.causalPUtrt_dir, "pu_tm.R"))       # Pure R
source(file.path(.causalPUtrt_dir, "pu_sarem.R"))    # Requires reticulate + Python sarpu
source(file.path(.causalPUtrt_dir, "pu_detm.R"))     # Requires PUEM R package

# ATE estimators
source(file.path(.causalPUtrt_dir, "ate_estimators.R"))
source(file.path(.causalPUtrt_dir, "ate_wrappers.R"))

message("causalPUtrt loaded. Available PU methods: TM (pure R), SAR-EM (Python), DETM (PUEM).")
message("  - TM:     ready to use")
message("  - SAR-EM: call init_sarem(virtualenv_path, sarpu_path) before use")
message("  - DETM:   requires PUEM package (library(PUEM))")
