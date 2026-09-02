# Tests for the shared weight-download confirmation gate used by both
# brulee_tab_icl() and brulee_chronos() (R/0_utils.R).

test_that("brulee_confirm_download errors when non-interactive", {
  testthat::local_mocked_bindings(
    is_interactive = function() FALSE,
    .package = "rlang"
  )

  expect_error(
    brulee:::brulee_confirm_download(
      label = "amazon/chronos-2",
      size = "500MB",
      fn = "brulee_chronos",
      root = tempdir(),
      hint = "Run {.fn brulee_chronos} in an interactive session to download them."
    ),
    "No cached .*amazon/chronos-2.* weights found"
  )
})

test_that("brulee_confirm_download aborts when the user declines", {
  testthat::local_mocked_bindings(
    is_interactive = function() TRUE,
    .package = "rlang"
  )
  testthat::local_mocked_bindings(
    menu = function(choices, ...) 2L,
    .package = "utils"
  )

  expect_error(
    suppressMessages(
      brulee:::brulee_confirm_download(
        label = "amazon/chronos-2",
        size = "500MB",
        fn = "brulee_chronos",
        root = tempdir(),
        hint = "Run {.fn brulee_chronos} in an interactive session to download them."
      )
    ),
    "Download declined"
  )
})

test_that("brulee_confirm_download returns TRUE when the user accepts", {
  testthat::local_mocked_bindings(
    is_interactive = function() TRUE,
    .package = "rlang"
  )
  testthat::local_mocked_bindings(
    menu = function(choices, ...) 1L,
    .package = "utils"
  )

  expect_true(
    suppressMessages(
      brulee:::brulee_confirm_download(
        label = "amazon/chronos-2",
        size = "500MB",
        fn = "brulee_chronos",
        root = tempdir(),
        hint = "Run {.fn brulee_chronos} in an interactive session to download them."
      )
    )
  )
})

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
