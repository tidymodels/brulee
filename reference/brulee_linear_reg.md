# Fit a linear regression model

`brulee_linear_reg()` fits a linear regression model.

## Usage

``` r
brulee_linear_reg(x, ...)

# Default S3 method
brulee_linear_reg(x, ...)

# S3 method for class 'data.frame'
brulee_linear_reg(
  x,
  y,
  epochs = 20L,
  penalty = 0.001,
  mixture = 0,
  validation = 0.1,
  optimizer = "LBFGS",
  learn_rate = 1,
  momentum = 0,
  batch_size = NULL,
  stop_iter = 5,
  verbose = FALSE,
  device = NULL,
  ...
)

# S3 method for class 'matrix'
brulee_linear_reg(
  x,
  y,
  epochs = 20L,
  penalty = 0.001,
  mixture = 0,
  validation = 0.1,
  optimizer = "LBFGS",
  learn_rate = 1,
  momentum = 0,
  batch_size = NULL,
  stop_iter = 5,
  verbose = FALSE,
  device = NULL,
  ...
)

# S3 method for class 'formula'
brulee_linear_reg(
  formula,
  data,
  epochs = 20L,
  penalty = 0.001,
  mixture = 0,
  validation = 0.1,
  optimizer = "LBFGS",
  learn_rate = 1,
  momentum = 0,
  batch_size = NULL,
  stop_iter = 5,
  verbose = FALSE,
  device = NULL,
  ...
)

# S3 method for class 'recipe'
brulee_linear_reg(
  x,
  data,
  epochs = 20L,
  penalty = 0.001,
  mixture = 0,
  validation = 0.1,
  optimizer = "LBFGS",
  learn_rate = 1,
  momentum = 0,
  batch_size = NULL,
  stop_iter = 5,
  verbose = FALSE,
  device = NULL,
  ...
)
```

## Arguments

- x:

  Depending on the context:

  - A **data frame** of predictors.

  - A **matrix** of predictors.

  - A **recipe** specifying a set of preprocessing steps created from
    [`recipes::recipe()`](https://recipes.tidymodels.org/reference/recipe.html).

  The predictor data should be standardized (e.g. centered or scaled).

- ...:

  Options to pass to the learning rate schedulers via
  [`set_learn_rate()`](https://brulee.tidymodels.org/reference/schedule_decay_time.md).
  For example, the `reduction` or `steps` arguments to
  [`schedule_step()`](https://brulee.tidymodels.org/reference/schedule_decay_time.md)
  could be passed here.

- y:

  When `x` is a **data frame** or **matrix**, `y` is the outcome
  specified as:

  - A **data frame** with 1 numeric column.

  - A **matrix** with 1 numeric column.

  - A numeric **vector**.

- epochs:

  An integer for the number of epochs of training.

- penalty:

  The amount of weight decay (i.e., L2 regularization).

- mixture:

  Proportion of Lasso Penalty (type: double, default: 0.0). A value of
  mixture = 1 corresponds to a pure lasso model, while mixture = 0
  indicates ridge regression (a.k.a weight decay). Must be zero for
  optimizers `"ADAMw"`, `"RMSprop"`, `"Adadelta"`.

- validation:

  The proportion of the data randomly assigned to a validation set.

- optimizer:

  The method used in the optimization procedure. Possible choices are
  `"SGD"`, `"ADAMw"`, `"Adadelta"`, `"Adagrad"`, `"RMSprop"`, and
  `"LBFGS"`. `"LBFGS"` is the only second-order method, does not use
  batches, and is the default.

- learn_rate:

  A positive number that controls the initial rapidity that the model
  moves along the descent path. Values around 0.1 or less are typical.

- momentum:

  A positive number usually on `[0.50, 0.99]` for the momentum parameter
  in gradient descent. (optimizers `"SGD"`, and `"RMSprop"` only,
  ignored otherwise).

- batch_size:

  An integer for the number of training set points in each batch.
  (`optimizer != "LBFGS"` only, ignored otherwise)

- stop_iter:

  A non-negative integer for how many iterations with no improvement
  before stopping.

- verbose:

  A logical that prints out the iteration history.

- device:

  A single character string for the device to train on (e.g., `"cpu"` or
  `"cuda"` for GPU). If `NULL`, the function will use the GPU if
  available, otherwise CPU. See
  [training_efficiency](https://brulee.tidymodels.org/reference/training_efficiency.md).

- formula:

  A formula specifying the outcome term(s) on the left-hand side, and
  the predictor term(s) on the right-hand side.

- data:

  When a **recipe** or **formula** is used, `data` is specified as:

  - A **data frame** containing both the predictors and the outcome.

## Value

A `brulee_linear_reg` object with elements:

- `models_obj`: a serialized raw vector for the torch module.

- `estimates`: a list of matrices with the model parameter estimates per
  epoch.

- `best_epoch`: an integer for the epoch with the smallest loss.

- `loss`: A vector of loss values (MSE) at each epoch.

- `dim`: A list of data dimensions.

- `y_stats`: A list of summary statistics for numeric outcomes.

- `parameters`: A list of some tuning parameter values.

- `blueprint`: The `hardhat` blueprint data.

## Details

This function fits a linear combination of coefficients and predictors
to model the numeric outcome. The training process optimizes the mean
squared error loss function.

The function internally standardizes the outcome data to have mean zero
and a standard deviation of one. The prediction function creates
predictions on the original scale.

By default, training halts when the validation loss increases for at
least `step_iter` iterations. If `validation = 0` the training set loss
is used.

The *predictors* data should all be numeric and encoded in the same
units (e.g. standardized to the same range or distribution). If there
are factor predictors, use a recipe or formula to create indicator
variables (or some other method) to make them numeric. Predictors should
be in the same units before training.

The model objects are saved for each epoch so that the number of epochs
can be efficiently tuned. Both the
[`coef()`](https://rdrr.io/r/stats/coef.html) and
[`predict()`](https://rdrr.io/r/stats/predict.html) methods for this
model have an `epoch` argument (which defaults to the epoch with the
best loss value).

The use of the L1 penalty (a.k.a. the lasso penalty) does *not* force
parameters to be strictly zero (as it does in packages such as glmnet).
The zeroing out of parameters is a specific feature the optimization
method used in those packages.

## See also

[`predict.brulee_linear_reg()`](https://brulee.tidymodels.org/reference/predict.brulee_linear_reg.md),
[`coef.brulee_linear_reg()`](https://brulee.tidymodels.org/reference/brulee-coefs.md),
[`autoplot.brulee_linear_reg()`](https://brulee.tidymodels.org/reference/brulee-autoplot.md)

## Examples

``` r
# \donttest{
if (torch::torch_is_installed()  & rlang::is_installed(c("recipes", "yardstick", "modeldata"))) {

 ## -----------------------------------------------------------------------------

 library(recipes)
 library(yardstick)

 data(ames, package = "modeldata")

 ames$Sale_Price <- log10(ames$Sale_Price)

 set.seed(122)
 in_train <- sample(1:nrow(ames), 2000)
 ames_train <- ames[ in_train,]
 ames_test  <- ames[-in_train,]


 # Using matrices
 set.seed(1)
 brulee_linear_reg(x = as.matrix(ames_train[, c("Longitude", "Latitude")]),
                    y = ames_train$Sale_Price,
                    penalty = 0.10, epochs = 1, batch_size = 64)

 # Using recipe
 library(recipes)

 ames_rec <-
  recipe(Sale_Price ~ Bldg_Type + Neighborhood + Year_Built + Gr_Liv_Area +
         Full_Bath + Year_Sold + Lot_Area + Central_Air + Longitude + Latitude,
         data = ames_train) |>
    # Transform some highly skewed predictors
    step_BoxCox(Lot_Area, Gr_Liv_Area) |>
    # Lump some rarely occurring categories into "other"
    step_other(Neighborhood, threshold = 0.05)  |>
    # Encode categorical predictors as binary.
    step_dummy(all_nominal_predictors(), one_hot = TRUE) |>
    # Add an interaction effect:
    step_interact(~ starts_with("Central_Air"):Year_Built) |>
    step_zv(all_predictors()) |>
    step_normalize(all_numeric_predictors())

 set.seed(2)
 fit <- brulee_linear_reg(ames_rec, data = ames_train, epochs = 5)
 fit

 autoplot(fit)

 library(ggplot2)

 predict(fit, ames_test) |>
   bind_cols(ames_test) |>
   ggplot(aes(x = .pred, y = Sale_Price)) +
   geom_abline(col = "green") +
   geom_point(alpha = .3) +
   lims(x = c(4, 6), y = c(4, 6)) +
   coord_fixed(ratio = 1)

 library(yardstick)
 predict(fit, ames_test) |>
   bind_cols(ames_test) |>
   rmse(Sale_Price, .pred)

 }
#> Warning: `batch_size` is only used for the SGD optimizer.
#> # A tibble: 1 × 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 rmse    standard      0.0821

# }
```
