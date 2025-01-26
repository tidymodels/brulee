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
