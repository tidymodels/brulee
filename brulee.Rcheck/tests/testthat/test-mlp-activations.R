test_that("activation functions", {
  skip_on_cran()
  skip_if(!torch::torch_is_installed())
  skip_if_not_installed("modeldata")

  # ------------------------------------------------------------------------------

  set.seed(1)
  df <- modeldata::sim_regression(500)

  acts <- brulee_activations()
  acts <- acts[acts != "linear"]
  # `rrelu_with_noise` is not implemented in torch's MPS backend
  # (genuine op-not-supported, unrelated to the MPS RNG init issue).
  # Skip the rrelu activation on MPS; it remains tested on CPU/CUDA.
  if (identical(brulee:::guess_brulee_device(NULL), "mps")) {
    acts <- acts[acts != "rrelu"]
  }

  for (i in acts) {
    expect_no_error(
      {
        set.seed(2)
        model <- brulee_mlp(
          outcome ~ .,
          data = df[1:400, ],
          activation = i,
          penalty = 0.1,
          learn_rate = 0.01,
          epochs = 50L,
          hidden_units = 20L,
          optimizer = "ADAMw",
          batch_size = 16,
        )
      }
    )

    r_sq <- cor(predict(model, df[401:500, -1])$.pred, df$outcome[401:500])^2
    # cat("act", i, "rsq:", signif(r_sq, 3), "\n")

    # These do very poorly on this problems
    pass <- c("tanhshrink", "log_sigmoid", "softplus")

    if (!(i %in% pass)) {
      expect_true(r_sq > 0.1)
    }
  }
})
