test_that("augment() for regression models", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  set.seed(1)
  reg_tr <- tibble::tibble(
    x1 = runif(200),
    x2 = runif(200),
    outcome = 3 + 2 * x1 + 3 * x2
  )

  set.seed(392)
  torch::torch_manual_seed(392)
  reg_fit <- brulee_linear_reg(outcome ~ ., reg_tr, epochs = 20, device = "cpu")

  # The outcome is in `new_data`, so residuals are added.
  aug <- augment(reg_fit, reg_tr)

  exp_str <-
    structure(
      list(
        .pred = numeric(0),
        .resid = numeric(0),
        x1 = numeric(0),
        x2 = numeric(0),
        outcome = numeric(0)
      ),
      row.names = integer(0),
      class = c("tbl_df", "tbl", "data.frame")
    )

  expect_equal(aug[0, ], exp_str)
  expect_equal(nrow(aug), nrow(reg_tr))
  expect_equal(aug$.pred, predict(reg_fit, reg_tr)$.pred)
  expect_equal(aug$.resid, reg_tr$outcome - aug$.pred)

  # Without the outcome column there are no residuals.
  no_y <- augment(reg_fit, reg_tr[, c("x1", "x2")])
  expect_named(no_y, c(".pred", "x1", "x2"))

  # Zero rows in, zero rows out.
  expect_equal(nrow(augment(reg_fit, reg_tr[0, ])), 0L)
})

test_that("augment() uses `.outcome` for x/y fits, so adds no residuals", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  set.seed(1)
  x <- matrix(runif(400), ncol = 2, dimnames = list(NULL, c("x1", "x2")))
  y <- 3 + 2 * x[, "x1"] + 3 * x[, "x2"]

  set.seed(392)
  torch::torch_manual_seed(392)
  fit <- brulee_linear_reg(x, y, epochs = 20, device = "cpu")

  # The blueprint records the outcome as ".outcome", which is not a column of
  # `new_data`, so residuals cannot be computed.
  expect_equal(brulee_outcome_name(fit), ".outcome")

  aug <- augment(fit, x)
  expect_named(aug, c(".pred", "x1", "x2"))
  expect_equal(nrow(aug), nrow(x))
})

test_that("augment() for binary classification", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())
  skip_if_not_installed("modeldata")

  data(penguins, package = "modeldata")
  penguins <- penguins[complete.cases(penguins), ]

  set.seed(392)
  torch::torch_manual_seed(392)
  cls_fit <- brulee_logistic_reg(
    sex ~ bill_length_mm + body_mass_g,
    penguins,
    epochs = 5,
    device = "cpu"
  )

  aug <- augment(cls_fit, penguins)

  expect_equal(
    names(aug)[1:3],
    c(".pred_class", ".pred_female", ".pred_male")
  )
  expect_equal(nrow(aug), nrow(penguins))
  expect_s3_class(aug$.pred_class, "factor")
  expect_equal(levels(aug$.pred_class), levels(penguins$sex))

  # No residuals for classification, even though the outcome is present.
  expect_false(".resid" %in% names(aug))

  # The columns agree with the corresponding `predict()` calls.
  expect_equal(
    aug$.pred_class,
    predict(cls_fit, penguins, type = "class")$.pred_class
  )
  expect_equal(
    aug$.pred_female,
    predict(cls_fit, penguins, type = "prob")$.pred_female
  )
})

test_that("augment() for multiclass classification", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())
  skip_if_not_installed("modeldata")

  data(penguins, package = "modeldata")
  penguins <- penguins[complete.cases(penguins), ]

  set.seed(392)
  torch::torch_manual_seed(392)
  cls_fit <- brulee_multinomial_reg(
    species ~ bill_length_mm + body_mass_g,
    penguins,
    epochs = 5,
    device = "cpu"
  )

  aug <- augment(cls_fit, penguins)

  expect_equal(
    names(aug)[1:4],
    c(".pred_class", ".pred_Adelie", ".pred_Chinstrap", ".pred_Gentoo")
  )
  expect_equal(nrow(aug), nrow(penguins))

  probs <- as.matrix(aug[, 2:4])
  expect_equal(unname(rowSums(probs)), rep(1, nrow(aug)), tolerance = 1e-5)
})

test_that("augment() passes `...` through to predict()", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  set.seed(1)
  reg_tr <- tibble::tibble(
    x1 = runif(200),
    x2 = runif(200),
    outcome = 3 + 2 * x1 + 3 * x2
  )

  set.seed(392)
  torch::torch_manual_seed(392)
  reg_fit <- brulee_mlp(outcome ~ ., reg_tr, epochs = 20, device = "cpu")

  expect_equal(
    augment(reg_fit, reg_tr, epoch = 3)$.pred,
    predict(reg_fit, reg_tr, epoch = 3)$.pred
  )
  # An early epoch differs from the best one, so the argument demonstrably
  # reaches `predict()`.
  expect_false(
    isTRUE(all.equal(
      augment(reg_fit, reg_tr)$.pred,
      augment(reg_fit, reg_tr, epoch = 1)$.pred
    ))
  )
})

test_that("augment() reaches both predict() calls for classification", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())
  skip_if_not_installed("modeldata")

  data(penguins, package = "modeldata")
  penguins <- penguins[complete.cases(penguins), ]

  set.seed(392)
  torch::torch_manual_seed(392)
  cls_fit <- brulee_mlp(
    species ~ bill_length_mm + body_mass_g,
    penguins,
    epochs = 20,
    device = "cpu"
  )

  aug <- augment(cls_fit, penguins, epoch = 2)

  expect_equal(
    aug$.pred_class,
    predict(cls_fit, penguins, type = "class", epoch = 2)$.pred_class
  )
  expect_equal(
    aug$.pred_Adelie,
    predict(cls_fit, penguins, type = "prob", epoch = 2)$.pred_Adelie
  )
})

test_that("augment() errors when the outcome column is not numeric", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  set.seed(1)
  reg_tr <- tibble::tibble(
    x1 = runif(200),
    x2 = runif(200),
    outcome = 3 + 2 * x1 + 3 * x2
  )

  set.seed(392)
  torch::torch_manual_seed(392)
  reg_fit <- brulee_linear_reg(outcome ~ ., reg_tr, epochs = 20, device = "cpu")

  bad <- reg_tr
  bad$outcome <- as.character(bad$outcome)

  expect_snapshot(augment(reg_fit, bad), error = TRUE)
})

test_that("augment() works for brulee_tab_icl, which has no `epoch`", {
  skip_if_no_tabicl_fixtures("engine_clf")

  f <- tabicl_load_fixture("engine_clf")
  meta <- tabicl_fixture_meta("engine_clf")
  tabicl_local_cache(f, meta)

  x_train <- as.data.frame(as.matrix(as.array(f$X_train)))
  y_train <- factor(as.integer(as.numeric(as.array(f$y_train))))
  x_test <- as.data.frame(as.matrix(as.array(f$X_test)))
  names(x_test) <- names(x_train)

  fit <- brulee_tab_icl(x_train, y_train, num_estimators = 1L)

  aug <- augment(fit, x_test)

  expect_equal(nrow(aug), nrow(x_test))
  expect_equal(names(aug)[1], ".pred_class")
  expect_equal(
    names(aug)[2:(1 + nlevels(y_train))],
    paste0(".pred_", levels(y_train))
  )
  expect_equal(
    aug$.pred_class,
    predict(fit, x_test, type = "class")$.pred_class
  )
})
