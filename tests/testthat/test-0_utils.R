# ------------------------------------------------------------------------------
# clamp_epoch() (R/0_utils.R)
#
# `estimates` stores epoch zero as its first element, so a list of length `n`
# holds epochs `0..(n - 1)`. Only `length()` is used, so plain empty lists stand
# in for real parameter sets.

test_that("clamp_epoch() leaves an in-range epoch alone", {
  estimates <- vector("list", 8L)

  expect_no_warning(epoch <- brulee:::clamp_epoch(3L, estimates))
  expect_equal(epoch, 3L)

  # Epoch zero is valid: it is the initial, pre-training parameters.
  expect_no_warning(epoch <- brulee:::clamp_epoch(0L, estimates))
  expect_equal(epoch, 0L)

  # The largest valid epoch is one less than the length of the list. This is the
  # boundary the old `epoch > length(estimates)` check got wrong.
  expect_no_warning(epoch <- brulee:::clamp_epoch(7L, estimates))
  expect_equal(epoch, 7L)
})

test_that("clamp_epoch() clamps an out-of-range epoch and warns", {
  estimates <- vector("list", 8L)

  # `epoch` equal to `length(estimates)` used to slip past the guard entirely,
  # emitting no warning and then erroring in `estimates[[epoch + 1]]`.
  expect_snapshot(epoch <- brulee:::clamp_epoch(8L, estimates))
  expect_equal(epoch, 7L)

  expect_snapshot(epoch <- brulee:::clamp_epoch(10L, estimates))
  expect_equal(epoch, 7L)
})

test_that("clamp_epoch() pluralizes the epoch count", {
  # Two elements means epochs 0 and 1, so "1 epoch" is the correct singular.
  expect_snapshot(epoch <- brulee:::clamp_epoch(5L, vector("list", 2L)))
  expect_equal(epoch, 1L)
})

# ------------------------------------------------------------------------------
# brulee_stratum_sizes() (R/0_utils.R)

test_that("brulee_stratum_sizes() allocates exactly `limit` rows", {
  expect_identical(
    brulee_stratum_sizes(c(40L, 30L, 20L, 10L), 20L),
    c(8L, 6L, 4L, 2L)
  )
  # At least one row per stratum, taken back from the largest.
  expect_identical(
    brulee_stratum_sizes(c(97L, 1L, 1L, 1L), 4L),
    c(1L, 1L, 1L, 1L)
  )
  # Rounding up overshoots; the excess comes off the largest stratum.
  expect_identical(brulee_stratum_sizes(c(5L, 5L, 5L), 4L), c(2L, 1L, 1L))
  expect_identical(sum(brulee_stratum_sizes(c(7L, 11L, 13L), 17L)), 17L)
})

# ------------------------------------------------------------------------------
# brulee_subsample_rows() (R/0_utils.R)

test_that("brulee_subsample_rows() keeps every row when under the limit", {
  outcome <- factor(rep(c("a", "b"), 50))
  expect_identical(brulee_subsample_rows(outcome, Inf), 1:100)
  expect_identical(brulee_subsample_rows(outcome, 100), 1:100)
  expect_identical(brulee_subsample_rows(outcome, 200), 1:100)
})

test_that("brulee_subsample_rows() keeps exactly `limit` rows, stratified", {
  set.seed(70913)
  four <- factor(rep(c("a", "b", "c", "d"), c(40, 30, 20, 10)))
  idx <- brulee_subsample_rows(four, 20)
  expect_length(idx, 20)
  expect_identical(as.vector(table(four[idx])), c(8L, 6L, 4L, 2L))

  rare <- factor(rep(c("a", "b", "c", "d"), c(97, 1, 1, 1)))
  expect_identical(
    as.vector(table(rare[brulee_subsample_rows(rare, 4)])),
    rep(1L, 4)
  )

  # Unused levels don't count as classes.
  unused <- factor(rep(c("a", "b"), 50), levels = c("a", "b", "c"))
  expect_length(brulee_subsample_rows(unused, 10), 10)

  numbers <- seq(0, 1, length.out = 101)
  idx <- brulee_subsample_rows(numbers, 40)
  expect_length(idx, 40)
  # Quartile strata: each quarter of the range keeps about a quarter.
  per_quarter <- as.vector(table(cut(
    numbers[idx],
    0:4 / 4,
    include.lowest = TRUE
  )))
  expect_all_true(abs(per_quarter - 10) <= 1)
  expect_length(brulee_subsample_rows(numbers, 2), 2)
})

test_that("brulee_subsample_rows() works with a constant outcome", {
  set.seed(70913)
  idx <- brulee_subsample_rows(rep(3, 50), 10)
  expect_length(idx, 10)
  expect_length(unique(idx), 10)
})

test_that("brulee_subsample_rows() can't drop a class", {
  four <- factor(rep(c("a", "b", "c", "d"), 10))
  expect_snapshot(brulee_subsample_rows(four, 3, call = NULL), error = TRUE)
})
