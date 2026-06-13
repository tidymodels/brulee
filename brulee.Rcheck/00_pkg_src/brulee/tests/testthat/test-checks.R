test_that("checking double vectors", {
  variable <- seq(0, 1, length = 3)
  expect_silent(check_number_decimal_vec(variable))
  expect_silent(check_number_decimal_vec(variable[1]))

  expect_snapshot(check_number_decimal_vec(letters), error = TRUE)

  variable <- NA_real_
  expect_snapshot(check_number_decimal_vec(variable), error = TRUE)
  expect_silent(check_number_decimal_vec(variable, allow_na = TRUE))

  variable <- 1L
  expect_snapshot(check_number_decimal_vec(variable), error = TRUE)
})

test_that("checking whole number vectors", {
  variable <- 1:2
  expect_silent(check_number_whole_vec(variable))
  expect_silent(check_number_whole_vec(variable[1]))

  variable <- seq(0, 1, length.out = 3)
  expect_snapshot(check_number_whole_vec(variable), error = TRUE)

  variable <- NA_integer_
  expect_snapshot(check_number_whole_vec(variable), error = TRUE)
  expect_silent(check_number_whole_vec(variable, allow_na = TRUE))
})

test_that("check_optimizer validates optimizer names", {
  expect_silent(brulee:::check_optimizer("SGD"))
  expect_silent(brulee:::check_optimizer("ADAMw"))
  expect_silent(brulee:::check_optimizer("Adadelta"))
  expect_silent(brulee:::check_optimizer("Adagrad"))
  expect_silent(brulee:::check_optimizer("RMSprop"))
  expect_silent(brulee:::check_optimizer("LBFGS"))

  x <- "adam"
  expect_snapshot(brulee:::check_optimizer(x), error = TRUE)
  x <- "nope"
  expect_snapshot(brulee:::check_optimizer(x), error = TRUE)
  x <- 123
  expect_snapshot(brulee:::check_optimizer(x), error = TRUE)
  x <- c("SGD", "LBFGS")
  expect_snapshot(brulee:::check_optimizer(x), error = TRUE)
})

test_that("check_classification_loss validates loss names", {
  expect_silent(brulee:::check_classification_loss("nll"))
  expect_silent(brulee:::check_classification_loss("focal"))

  x <- "mse"
  expect_snapshot(brulee:::check_classification_loss(x), error = TRUE)
  x <- 123
  expect_snapshot(brulee:::check_classification_loss(x), error = TRUE)
})

test_that("check_integer validates integers with bounds", {
  x <- 5L
  expect_silent(brulee:::check_integer(x, single = TRUE, x_min = 1))

  x <- 0L
  expect_snapshot(
    brulee:::check_integer(x, single = TRUE, x_min = 1),
    error = TRUE
  )

  x <- "a"
  expect_snapshot(brulee:::check_integer(x, single = TRUE), error = TRUE)

  x <- c(1L, 2L, 3L)
  expect_silent(brulee:::check_integer(x, single = FALSE, x_min = 1))
})

test_that("check_double validates doubles with exclusive bounds", {
  x <- 0.5
  expect_silent(brulee:::check_double(x, single = TRUE, x_min = 0, x_max = 1))

  # Exclusive lower bound
  x <- 0.0
  expect_snapshot(
    brulee:::check_double(x, single = TRUE, x_min = 0, incl = c(FALSE, TRUE)),
    error = TRUE
  )

  # Exclusive upper bound
  x <- 1.0
  expect_snapshot(
    brulee:::check_double(
      x,
      single = TRUE,
      x_min = 0,
      x_max = 1,
      incl = c(TRUE, FALSE)
    ),
    error = TRUE
  )

  # Within exclusive bounds is fine
  x <- 0.5
  expect_silent(
    brulee:::check_double(
      x,
      single = TRUE,
      x_min = 0,
      x_max = 1,
      incl = c(FALSE, FALSE)
    )
  )
})
