
test_that("tunable values", {
 expect_snapshot(brulee:::tunable.brulee_linear_reg(1)$call_info)
 expect_snapshot(brulee:::tunable.brulee_logistic_reg(1)$call_info)
 expect_snapshot(brulee:::tunable.brulee_multinomial_reg(1)$call_info)
 expect_snapshot(brulee:::tunable.brulee_mlp(1)$call_info)
 expect_snapshot(brulee:::tunable.brulee_mlp_two_layer(1)$call_info)
})
