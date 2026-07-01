# Keep the TabICL attach hook (R/aaa.R) from downloading weights over the network
# during the package's own tests and R CMD check. End users still get the
# automatic download; the package's checks opt out.
options(brulee.tabicl_autodownload = FALSE)

# Pin torch to a single intra-op thread for the whole suite. libtorch only honors
# this before any parallel work has started, so it must happen here (in setup,
# before the first tensor op) rather than inside individual tests. Single-threaded
# reductions are deterministic within a platform, which the gradient-clipping
# overflow tests (test-autoint.R / test-saint.R) rely on to behave consistently.
if (rlang::is_installed("torch")) {
  try(torch::torch_set_num_threads(1), silent = TRUE)
}
