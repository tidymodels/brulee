# Numerical-parity harness: R side.
#
# Confirms R can consume every artifact the Python converter + golden dumper
# produce: the config, all state_dict tensors, and the per-stage golden tensors.
# This is the "weights load, names all matched, no skips" gate for step 1 of the
# TabICL port. It does NOT yet build the R nn_module tree (steps 3-5); it proves
# the loading infrastructure and inventories the parameter names that the
# name-map work will target.
#
# Usage:  Rscript dev/tabicl/harness.R [classifier|regressor]

suppressPackageStartupMessages({
  library(torch)
  library(safetensors)
})

args <- commandArgs(trailingOnly = TRUE)
kind <- if (length(args) >= 1) args[[1]] else "classifier"
stopifnot(kind %in% c("classifier", "regressor"))

art <- file.path(
  normalizePath(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1]))),
  "artifacts",
  kind
)
if (!dir.exists(art)) {
  art <- file.path("dev", "tabicl", "artifacts", kind)
}

cli_h <- function(x) cat("\n== ", x, " ==\n", sep = "")

# --- config -----------------------------------------------------------------
cli_h(paste("config:", kind))
config <- jsonlite::fromJSON(file.path(art, "config.json"))
cat("config keys:", length(config), "\n")
cat("max_classes =", config$max_classes, " num_quantiles =", config$num_quantiles, "\n")

# --- state_dict -------------------------------------------------------------
cli_h("state_dict (model.safetensors)")
tensors <- safe_load_file(file.path(art, "model.safetensors"), framework = "torch")
expected <- readLines(file.path(art, "state_dict_keys.txt"))
expected_names <- vapply(strsplit(expected, "\t"), `[`, character(1), 1L)

n_loaded <- length(tensors)
cat("tensors loaded:", n_loaded, " expected:", length(expected_names), "\n")

missing_in_r <- setdiff(expected_names, names(tensors))
extra_in_r <- setdiff(names(tensors), expected_names)
stopifnot(length(missing_in_r) == 0L, length(extra_in_r) == 0L)

# Every tensor must materialize as a torch_tensor with a defined shape (no skips).
bad <- character()
for (nm in names(tensors)) {
  t <- tensors[[nm]]
  if (!inherits(t, "torch_tensor") || any(is.na(dim(t)))) {
    bad <- c(bad, nm)
  }
}
stopifnot(length(bad) == 0L)
cat("all tensors materialized as torch_tensor with defined shapes: OK\n")

# Top-level module inventory (the three stages + any heads).
prefixes <- sub("\\..*$", "", names(tensors))
cat("top-level prefixes:\n")
print(table(prefixes))

# --- golden fixtures --------------------------------------------------------
cli_h("golden fixtures")
golden_dir <- file.path(art, "golden")
meta <- jsonlite::fromJSON(file.path(golden_dir, "meta.json"))
inputs <- safe_load_file(file.path(golden_dir, "inputs.safetensors"), framework = "torch")
stages <- safe_load_file(file.path(golden_dir, "stage_outputs.safetensors"), framework = "torch")

check_shape <- function(t, expected_shape, label) {
  got <- dim(t)
  ok <- identical(as.integer(got), as.integer(expected_shape))
  cat(sprintf("  %-14s R=[%s] py=[%s] %s\n",
    label,
    paste(got, collapse = ","),
    paste(expected_shape, collapse = ","),
    if (ok) "OK" else "MISMATCH"))
  stopifnot(ok)
}

cat("seed (python):", meta$seed, "\n")
check_shape(inputs$X, meta$shapes$X, "X")
check_shape(inputs$y_train, meta$shapes$y_train, "y_train")
check_shape(stages$col_embed, meta$shapes$col_embed, "col_embed")
check_shape(stages$row_interact, meta$shapes$row_interact, "row_interact")
check_shape(stages$icl_out, meta$shapes$icl_out, "icl_out")

cli_h("PASS")
cat("step-1 harness OK for", kind, "\n")
