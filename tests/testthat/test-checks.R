test_that("checking double vectors", {
 library(rlang)

 variable <- seq(0, 1)
 expect_silent(brulee:::check_number_decimal_vec(variable))
 expect_silent(brulee:::check_number_decimal_vec(variable[1]))

 expect_snapshot(brulee:::check_number_decimal_vec(letters), error = TRUE)

 variable <- NA_real_
 expect_snapshot(brulee:::check_number_decimal_vec(variable), error = TRUE)
 expect_silent(brulee:::check_number_decimal_vec(variable, allow_na = TRUE))

 variable <- 1L
 expect_silent(brulee:::check_number_decimal_vec(variable))

})


test_that("checking whole number vectors", {
 library(rlang)

 variable <- 1:2
 expect_silent(brulee:::check_number_whole_vec(variable))
 expect_silent(brulee:::check_number_whole_vec(variable[1]))

 variable <- seq(0, 1, length.out = 3)
 expect_snapshot(brulee:::check_number_whole_vec(variable), error = TRUE)

 variable <- NA_integer_
 expect_snapshot(brulee:::check_number_whole_vec(variable), error = TRUE)
 expect_silent(brulee:::check_number_whole_vec(variable, allow_na = TRUE))

})
