# Pin torch to a single intra-op thread for the whole suite. libtorch only honors
# this before any parallel work has started, so it must happen here (in setup,
# before the first tensor op) rather than inside individual tests. Single-threaded
# reductions are deterministic within a platform, which the gradient-clipping
# overflow tests (test-autoint.R / test-saint.R) rely on to behave consistently.
if (rlang::is_installed("torch")) {
  try(torch::torch_set_num_threads(1), silent = TRUE)
}
