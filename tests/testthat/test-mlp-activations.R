
test_that("activation functions", {
 skip_if(!torch::torch_is_installed())
 skip_if_not_installed("modeldata")

 # ------------------------------------------------------------------------------

 set.seed(1)
 df <- modeldata::sim_regression(500)

 acts <- brulee_activations()
 acts <- acts[acts != "linear"]

 for (i in acts) {
  expect_error({
   set.seed(2)
   model <- brulee_mlp(outcome ~ ., data = df[1:400,],
                       activation = i,
                       learn_rate = 0.05,
                       hidden_units = 10L)

  },
  regex = NA
  )

  r_sq <- cor(predict(model, df[401:500, -1])$.pred, df$outcome[401:500])^2

    # These do very poorly on this problems
  pass <- c("tanhshrink")

  if (!(i %in% pass)) {
   expect_true(r_sq > 0.1)
  }
 }

})

