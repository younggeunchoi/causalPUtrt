# test_clean_install.R
# Simulates a clean environment by using a temporary library path.
# Run from the repo root:
#   R_LIBS_USER=$(mktemp -d) Rscript tests/test_clean_install.R
#
# This installs MASS into the temp dir, runs the examples, and reports pass/fail.

cat("=== Clean install test ===\n\n")

# Use only the temp library (strips existing user libs)
tmp_lib <- Sys.getenv("R_LIBS_USER")
if (tmp_lib == "" || tmp_lib == Sys.getenv("HOME")) {
  tmp_lib <- tempdir()
}
.libPaths(c(tmp_lib, .Library))
cat("Library paths:\n")
print(.libPaths())
cat("\n")

# Step 1: Run one-sample example
cat("--- Running example_onesample.R ---\n")
tryCatch({
  source("examples/example_onesample.R")
  cat("  PASS\n\n")
}, error = function(e) {
  cat("  FAIL:", conditionMessage(e), "\n\n")
  quit(status = 1)
})

# Step 2: Run two-sample example
cat("--- Running example_twosample.R ---\n")
tryCatch({
  source("examples/example_twosample.R")
  cat("  PASS\n\n")
}, error = function(e) {
  cat("  FAIL:", conditionMessage(e), "\n\n")
  quit(status = 1)
})

cat("=== All clean install tests passed ===\n")
