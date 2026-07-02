# Fit a logistic regression model

`brulee_logistic_reg()` fits a model.

## Usage

``` r
brulee_logistic_reg(x, ...)

# Default S3 method
brulee_logistic_reg(x, ...)

# S3 method for class 'data.frame'
brulee_logistic_reg(
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
  class_weights = NULL,
  stop_iter = 5,
  verbose = FALSE,
  device = NULL,
  ...
)

# S3 method for class 'matrix'
brulee_logistic_reg(
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
  class_weights = NULL,
  stop_iter = 5,
  verbose = FALSE,
  device = NULL,
  ...
)

# S3 method for class 'formula'
brulee_logistic_reg(
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
  class_weights = NULL,
  stop_iter = 5,
  verbose = FALSE,
  device = NULL,
  ...
)

# S3 method for class 'recipe'
brulee_logistic_reg(
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
  class_weights = NULL,
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

  - A **data frame** with 1 factor column (with two levels).

  - A **matrix** with 1 factor column (with two levels).

  - A factor **vector** (with two levels).

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

- class_weights:

  Numeric class weights (classification only). The value can be:

  - A named numeric vector (in any order) where the names are the
    outcome factor levels.

  - An unnamed numeric vector assumed to be in the same order as the
    outcome factor levels.

  - A single numeric value for the least frequent class in the training
    data and all other classes receive a weight of one.

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

A `brulee_logistic_reg` object with elements:

- `models_obj`: a serialized raw vector for the torch module.

- `estimates`: a list of matrices with the model parameter estimates per
  epoch. The first element is epoch zero (the randomly initialized
  parameters before training), so the list has `epochs + 1` elements.

- `best_epoch`: an integer for the epoch with the smallest loss. Since
  `estimates` and `loss` include epoch zero, this epoch's values are at
  position `best_epoch + 1` in those objects.

- `loss`: A vector of loss values (MSE for regression, negative log-
  likelihood for classification) at each epoch, starting with epoch
  zero.

- `dim`: A list of data dimensions.

- `parameters`: A list of some tuning parameter values.

- `blueprint`: The `hardhat` blueprint data.

## Details

This function fits a linear combination of coefficients and predictors
to model the log odds of the classes. The training process optimizes the
cross-entropy loss function (a.k.a Bernoulli loss).

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

[`predict.brulee_logistic_reg()`](https://brulee.tidymodels.org/reference/predict.brulee_logistic_reg.md),
[`coef.brulee_logistic_reg()`](https://brulee.tidymodels.org/reference/brulee-coefs.md),
[`autoplot.brulee_logistic_reg()`](https://brulee.tidymodels.org/reference/brulee-autoplot.md)

## Examples

``` r
# \donttest{
if (torch::torch_is_installed() && rlang::is_installed(c("recipes", "yardstick", "modeldata"))) {

 library(recipes)
 library(yardstick)

 ## -----------------------------------------------------------------------------
 # increase # epochs to get better results

 data(cells, package = "modeldata")

 cells$case <- NULL

 set.seed(122)
 in_train <- sample(seq_len(nrow(cells)), 1000)
 cells_train <- cells[ in_train,]
 cells_test  <- cells[-in_train,]

 # Using matrices
 set.seed(1)
 brulee_logistic_reg(x = as.matrix(cells_train[, c("fiber_width_ch_1", "width_ch_1")]),
                      y = cells_train$class,
                      penalty = 0.10, epochs = 3)

 # Using recipe
 library(recipes)

 cells_rec <-
  recipe(class ~ ., data = cells_train) |>
  # Transform some highly skewed predictors
  step_YeoJohnson(all_numeric_predictors()) |>
  step_normalize(all_numeric_predictors()) |>
  step_pca(all_numeric_predictors(), num_comp = 10)

 set.seed(2)
 fit <- brulee_logistic_reg(cells_rec, data = cells_train,
                             penalty = 0.01, epochs = 5)
 fit

 autoplot(fit)

 library(yardstick)
 predict(fit, cells_test, type = "prob") |>
  bind_cols(cells_test) |>
  roc_auc(class, .pred_PS)
}
#> # A tibble: 1 × 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 roc_auc binary         0.867
# }
```
