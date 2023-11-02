library(purrr)

test_that("scheduling functions", {

 x <- 0:100

 # ------------------------------------------------------------------------------

 expect_equal(
  map_dbl(x, schedule_decay_expo),
  0.1 * exp(-x)
 )

 expect_equal(
  map_dbl(x, schedule_decay_expo, initial = 1/3, decay = 7/8),
  1 / 3 * exp(-7 / 8 * x)
 )

 expect_snapshot_error(schedule_decay_expo(1, initial = -1))
 expect_snapshot_error(schedule_decay_expo(1, decay = -1))

 # ------------------------------------------------------------------------------

 expect_equal(
  map_dbl(x, schedule_decay_time),
  0.1 / (1 + x)
 )

 expect_equal(
  map_dbl(x, schedule_decay_time, initial = 1/3, decay = 7/8),
  1 / 3 / (1 + 7 / 8 * x)
 )

 expect_snapshot_error(schedule_decay_time(1, initial = -1))
 expect_snapshot_error(schedule_decay_time(1, decay = -1))

 # ------------------------------------------------------------------------------

 expect_equal(
  map_dbl(x, schedule_step),
  0.1 * (1 / 2) ^ floor(x / 5)
 )

 expect_equal(
  map_dbl(x, schedule_step, initial = 1/3, reduction = 7/8, steps = 3),
  1 / 3 * (7 / 8) ^ floor(x / 3)
 )

 expect_snapshot_error(schedule_step(1, initial = -1))
 expect_snapshot_error(schedule_step(1, reduction = -1))
 expect_snapshot_error(schedule_step(1, steps = -1))

 # ------------------------------------------------------------------------------

 expect_true( all(map_dbl(x[x %% 10 == 0], schedule_cyclic) == 0.001) )

 inc <- 0.0198
 expect_equal(
  abs(diff(map_dbl(x, schedule_cyclic))),
  rep(inc, 100),
  tolerance = 0.001
 )

 expect_equal(
  sign(diff(map_dbl(x, schedule_cyclic))),
  rep(rep(c(1, -1), each = 5), times = 10),
  tolerance = 0.001
 )

 expect_true( all(map_dbl(x[x %% 20 == 0], schedule_cyclic, step_size = 10) == 0.001) )


 expect_snapshot_error(schedule_cyclic(1, step_size = -1))
 expect_snapshot_error(schedule_cyclic(1, largest = -1))

 # ------------------------------------------------------------------------------

 expect_equal(set_learn_rate(.x, 1, type = "none"), 1)
 expect_equal(set_learn_rate(.x, 0.01, type = "none", potato = 1), .01)

 expect_equal(
  map_dbl(x, schedule_decay_time, initial = 1/3, decay = 7/8),
  map_dbl(x, ~ set_learn_rate(.x, 0.1, "decay_time", initial = 1/3, decay = 7/8))
 )

 expect_equal(
  map_dbl(x, schedule_decay_expo, initial = 1/3, decay = 7/8),
  map_dbl(x, ~ set_learn_rate(.x, 0.1, "decay_expo", initial = 1/3, decay = 7/8))
 )

 expect_equal(
  map_dbl(x, schedule_step, initial = 1/3, reduction = 7/8, steps = 3),
  map_dbl(x, ~ set_learn_rate(.x, 0.1, "step", initial = 1/3, reduction = 7/8, steps = 3))
 )

 expect_snapshot_error(set_learn_rate(1, 1, type = "decay_time", initial = -1))
 expect_snapshot_error(set_learn_rate(1, 1, type = "random"))

})
