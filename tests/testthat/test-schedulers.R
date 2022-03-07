library(purrr)

test_that("scheduling functions", {
 x <- 0:100

 # ------------------------------------------------------------------------------

 expect_equal(
  map_dbl(x, learn_rate_decay_expo),
  0.1 * exp(-x)
 )

 expect_equal(
  map_dbl(x, learn_rate_decay_expo, initial = 1/3, decay = 7/8),
  1 / 3 * exp(-7 / 8 * x)
 )

 expect_snapshot_error(learn_rate_decay_expo(1, initial = -1))
 expect_snapshot_error(learn_rate_decay_expo(1, decay = -1))

 # ------------------------------------------------------------------------------

 expect_equal(
  map_dbl(x, learn_rate_decay_time),
  0.1 / (1 + x)
 )

 expect_equal(
  map_dbl(x, learn_rate_decay_time, initial = 1/3, decay = 7/8),
  1 / 3 / (1 + 7 / 8 * x)
 )

 expect_snapshot_error(learn_rate_decay_time(1, initial = -1))
 expect_snapshot_error(learn_rate_decay_time(1, decay = -1))

 # ------------------------------------------------------------------------------

 expect_equal(
  map_dbl(x, learn_rate_step),
  0.1 * (1 / 2) ^ floor(x / 5)
 )

 expect_equal(
  map_dbl(x, learn_rate_step, initial = 1/3, reduction = 7/8, steps = 3),
  1 / 3 * (7 / 8) ^ floor(x / 3)
 )

 expect_snapshot_error(learn_rate_step(1, initial = -1))
 expect_snapshot_error(learn_rate_step(1, reduction = -1))
 expect_snapshot_error(learn_rate_step(1, steps = -1))


 # ------------------------------------------------------------------------------

 expect_equal(
  map_dbl(x, learn_rate_constant),
  rep(0.1, length(x))
 )

 expect_equal(
  map_dbl(x, learn_rate_constant, initial = 1/3),
  rep(1 / 3, length(x))
 )

 expect_snapshot_error(learn_rate_constant(1, initial = -1))

 # ------------------------------------------------------------------------------

 expect_equal(
  map_dbl(x, learn_rate_decay_time, initial = 1/3, decay = 7/8),
  map_dbl(x, ~ set_learn_rate(.x, "decay_time", initial = 1/3, decay = 7/8))
 )

 expect_equal(
  map_dbl(x, learn_rate_decay_expo, initial = 1/3, decay = 7/8),
  map_dbl(x, ~ set_learn_rate(.x, "decay_expo", initial = 1/3, decay = 7/8))
 )

 expect_equal(
  map_dbl(x, learn_rate_step, initial = 1/3, reduction = 7/8, steps = 3),
  map_dbl(x, ~ set_learn_rate(.x, "step", initial = 1/3, reduction = 7/8, steps = 3))
 )

 expect_equal(
  map_dbl(x, learn_rate_constant, initial = 1/3),
  map_dbl(x, ~ set_learn_rate(.x, "constant", initial = 1/3))
 )

 expect_snapshot_error(set_learn_rate(1, initial = -1))

})
