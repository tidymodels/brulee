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
