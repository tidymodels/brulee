# Fit a multinomial regression model

`brulee_multinomial_reg()` fits a model.

## Usage

``` r
brulee_multinomial_reg(x, ...)

# Default S3 method
brulee_multinomial_reg(x, ...)

# S3 method for class 'data.frame'
brulee_multinomial_reg(
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
brulee_multinomial_reg(
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
brulee_multinomial_reg(
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
brulee_multinomial_reg(
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
  [`set_learn_rate()`](https://brulee.tidymodels.org/dev/reference/schedule_decay_time.md).
  For example, the `reduction` or `steps` arguments to
  [`schedule_step()`](https://brulee.tidymodels.org/dev/reference/schedule_decay_time.md)
  could be passed here.

- y:

  When `x` is a **data frame** or **matrix**, `y` is the outcome
  specified as:

  - A **data frame** with 1 factor column (with three or more levels).

  - A **matrix** with 1 factor column (with three or more levels).

  - A factor **vector** (with three or more levels).

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
  [training_efficiency](https://brulee.tidymodels.org/dev/reference/training_efficiency.md).

- formula:

  A formula specifying the outcome term(s) on the left-hand side, and
  the predictor term(s) on the right-hand side.

- data:

  When a **recipe** or **formula** is used, `data` is specified as:

  - A **data frame** containing both the predictors and the outcome.

## Value

A `brulee_multinomial_reg` object with elements:

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
to model the log of the class probabilities. The training process
optimizes the cross-entropy loss function.

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

[`predict.brulee_multinomial_reg()`](https://brulee.tidymodels.org/dev/reference/predict.brulee_multinomial_reg.md),
[`coef.brulee_multinomial_reg()`](https://brulee.tidymodels.org/dev/reference/brulee-coefs.md),
[`autoplot.brulee_multinomial_reg()`](https://brulee.tidymodels.org/dev/reference/brulee-autoplot.md)

## Examples

``` r
# \donttest{
if (torch::torch_is_installed() && rlang::is_installed(c("recipes", "yardstick", "modeldata"))) {

  library(recipes)
  library(yardstick)

  data(penguins, package = "modeldata")

  penguins <- penguins |> na.omit()

  set.seed(122)
  in_train <- sample(seq_len(nrow(penguins)), 200)
  penguins_train <- penguins[ in_train,]
  penguins_test  <- penguins[-in_train,]

  rec <- recipe(island ~ ., data = penguins_train) |>
    step_dummy(species, sex) |>
    step_normalize(all_predictors())

  set.seed(3)
  fit <- brulee_multinomial_reg(rec, data = penguins_train, epochs = 5)
  fit

  predict(fit, penguins_test) |>
    bind_cols(penguins_test) |>
    conf_mat(island, .pred_class)
}
#>            Truth
#> Prediction  Biscoe Dream Torgersen
#>   Biscoe        49     2         3
#>   Dream         11    38         6
#>   Torgersen      9     8         7
# }
```
