# Fit neural networks

`brulee_mlp()` fits neural network models. Multiple layers can be used.
For working with two-layer networks in tidymodels,
`brulee_mlp_two_layer()` can be helpful for specifying tuning parameters
as scalars.

## Usage

``` r
brulee_mlp(x, ...)

# Default S3 method
brulee_mlp(x, ...)

# S3 method for class 'data.frame'
brulee_mlp(
  x,
  y,
  epochs = 100L,
  hidden_units = 3L,
  activation = "relu",
  penalty = 0.001,
  mixture = 0,
  dropout = 0,
  validation = 0.1,
  optimizer = "LBFGS",
  learn_rate = 0.01,
  rate_schedule = "none",
  momentum = 0,
  batch_size = NULL,
  class_weights = NULL,
  stop_iter = 5,
  grad_value_clip = 5,
  grad_norm_clip = 5,
  verbose = FALSE,
  ...
)

# S3 method for class 'matrix'
brulee_mlp(
  x,
  y,
  epochs = 100L,
  hidden_units = 3L,
  activation = "relu",
  penalty = 0.001,
  mixture = 0,
  dropout = 0,
  validation = 0.1,
  optimizer = "LBFGS",
  learn_rate = 0.01,
  rate_schedule = "none",
  momentum = 0,
  batch_size = NULL,
  class_weights = NULL,
  stop_iter = 5,
  grad_value_clip = 5,
  grad_norm_clip = 5,
  verbose = FALSE,
  ...
)

# S3 method for class 'formula'
brulee_mlp(
  formula,
  data,
  epochs = 100L,
  hidden_units = 3L,
  activation = "relu",
  penalty = 0.001,
  mixture = 0,
  dropout = 0,
  validation = 0.1,
  optimizer = "LBFGS",
  learn_rate = 0.01,
  rate_schedule = "none",
  momentum = 0,
  batch_size = NULL,
  class_weights = NULL,
  stop_iter = 5,
  grad_value_clip = 5,
  grad_norm_clip = 5,
  verbose = FALSE,
  ...
)

# S3 method for class 'recipe'
brulee_mlp(
  x,
  data,
  epochs = 100L,
  hidden_units = 3L,
  activation = "relu",
  penalty = 0.001,
  mixture = 0,
  dropout = 0,
  validation = 0.1,
  optimizer = "LBFGS",
  learn_rate = 0.01,
  rate_schedule = "none",
  momentum = 0,
  batch_size = NULL,
  class_weights = NULL,
  stop_iter = 5,
  grad_value_clip = 5,
  grad_norm_clip = 5,
  verbose = FALSE,
  ...
)

brulee_mlp_two_layer(x, ...)

# Default S3 method
brulee_mlp_two_layer(x, ...)

# S3 method for class 'data.frame'
brulee_mlp_two_layer(
  x,
  y,
  epochs = 100L,
  hidden_units = 3L,
  hidden_units_2 = 3L,
  activation = "relu",
  activation_2 = "relu",
  penalty = 0.001,
  mixture = 0,
  dropout = 0,
  validation = 0.1,
  optimizer = "LBFGS",
  learn_rate = 0.01,
  rate_schedule = "none",
  momentum = 0,
  batch_size = NULL,
  class_weights = NULL,
  stop_iter = 5,
  grad_value_clip = 5,
  grad_norm_clip = 5,
  verbose = FALSE,
  ...
)

# S3 method for class 'matrix'
brulee_mlp_two_layer(
  x,
  y,
  epochs = 100L,
  hidden_units = 3L,
  hidden_units_2 = 3L,
  activation = "relu",
  activation_2 = "relu",
  penalty = 0.001,
  mixture = 0,
  dropout = 0,
  validation = 0.1,
  optimizer = "LBFGS",
  learn_rate = 0.01,
  rate_schedule = "none",
  momentum = 0,
  batch_size = NULL,
  class_weights = NULL,
  stop_iter = 5,
  grad_value_clip = 5,
  grad_norm_clip = 5,
  verbose = FALSE,
  ...
)

# S3 method for class 'formula'
brulee_mlp_two_layer(
  formula,
  data,
  epochs = 100L,
  hidden_units = 3L,
  hidden_units_2 = 3L,
  activation = "relu",
  activation_2 = "relu",
  penalty = 0.001,
  mixture = 0,
  dropout = 0,
  validation = 0.1,
  optimizer = "LBFGS",
  learn_rate = 0.01,
  rate_schedule = "none",
  momentum = 0,
  batch_size = NULL,
  class_weights = NULL,
  stop_iter = 5,
  grad_value_clip = 5,
  grad_norm_clip = 5,
  verbose = FALSE,
  ...
)

# S3 method for class 'recipe'
brulee_mlp_two_layer(
  x,
  data,
  epochs = 100L,
  hidden_units = 3L,
  hidden_units_2 = 3L,
  activation = "relu",
  activation_2 = "relu",
  penalty = 0.001,
  mixture = 0,
  dropout = 0,
  validation = 0.1,
  optimizer = "LBFGS",
  learn_rate = 0.01,
  rate_schedule = "none",
  momentum = 0,
  batch_size = NULL,
  class_weights = NULL,
  stop_iter = 5,
  grad_value_clip = 5,
  grad_norm_clip = 5,
  verbose = FALSE,
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

  - A **data frame** with 1 column (numeric or factor).

  - A **matrix** with numeric column (numeric or factor).

  - A **vector** (numeric or factor).

- epochs:

  An integer for the number of epochs of training.

- hidden_units:

  An integer for the number of hidden units, or a vector of integers. If
  a vector of integers, the model will have `length(hidden_units)`
  layers each with `hidden_units[i]` hidden units.

- activation:

  A character vector for the activation function (such as "relu",
  "tanh", "sigmoid", and so on). See
  [`brulee_activations()`](https://brulee.tidymodels.org/dev/reference/brulee_activations.md)
  for a list of possible values. If `hidden_units` is a vector,
  `activation` can be a character vector with length equals to
  `length(hidden_units)` specifying the activation for each hidden
  layer.

- penalty:

  The amount of weight decay (i.e., L2 regularization).

- mixture:

  Proportion of Lasso Penalty (type: double, default: 0.0). A value of
  mixture = 1 corresponds to a pure lasso model, while mixture = 0
  indicates ridge regression (a.k.a weight decay). Must be zero for
  optimizers `"ADAMw"`, `"RMSprop"`, `"Adadelta"`.

- dropout:

  The proportion of parameters set to zero.

- validation:

  The proportion of the data randomly assigned to a validation set.

- optimizer:

  The method used in the optimization procedure. Possible choices are
  `"SGD"`, `"ADAMw"`, `"Adadelta"`, `"Adagrad"`, `"RMSprop"`, and
  `"LBFGS"`. `"LBFGS"` is the only second-order method and does not use
  batches.

- learn_rate:

  A positive number that controls the initial rapidity that the model
  moves along the descent path. Values around 0.1 or less are typical.

- rate_schedule:

  A single character value for how the learning rate should change as
  the optimization proceeds. Possible values are `"none"` (the default),
  `"decay_time"`, `"decay_expo"`, `"cyclic"` and `"step"`. See
  [`schedule_decay_time()`](https://brulee.tidymodels.org/dev/reference/schedule_decay_time.md)
  for more details.

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

- grad_norm_clip, grad_value_clip:

  Two numeric values, possibly `Inf`, that prevents the gradient's
  values or norm(s) from exceeding the specified value. This can be
  helpful if training stops early with the message that
  `"Loss is NaN at epoch x Training is stopped."`

- verbose:

  A logical that prints out the iteration history.

- formula:

  A formula specifying the outcome term(s) on the left-hand side, and
  the predictor term(s) on the right-hand side.

- data:

  When a **recipe** or **formula** is used, `data` is specified as:

  - A **data frame** containing both the predictors and the outcome.

- hidden_units_2:

  An integer for the number of hidden units for a second layer.

- activation_2:

  A character vector for the activation function for a second layer.

## Value

A `brulee_mlp` object with elements:

- `models_obj`: a serialized raw vector for the torch module.

- `estimates`: a list of matrices with the model parameter estimates per
  epoch.

- `best_epoch`: an integer for the epoch with the smallest loss.

- `loss`: A vector of loss values (MSE for regression, negative log-
  likelihood for classification) at each epoch.

- `dim`: A list of data dimensions.

- `y_stats`: A list of summary statistics for numeric outcomes.

- `parameters`: A list of some tuning parameter values.

- `blueprint`: The `hardhat` blueprint data.

## Details

This function fits feed-forward neural network models for regression
(when the outcome is a number) or classification (a factor). For
regression, the mean squared error is optimized and cross-entropy is the
loss function for classification.

When the outcome is a number, the function internally standardizes the
outcome data to have mean zero and a standard deviation of one. The
prediction function creates predictions on the original scale.

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

### Learning Rates

The learning rate can be set to constant (the default) or dynamically
set via a learning rate scheduler (via the `rate_schedule`). Using
`rate_schedule = 'none'` uses the `learn_rate` argument. Otherwise, any
arguments to the schedulers can be passed via `...`.

## References

adagrad (adaptive gradient algorithm): Duchi, J., Hazan, E., & Singer,
Y. (2011). Adaptive subgradient methods for online learning and
stochastic optimization. *Journal of machine learning research*, 12(7).

adadelta: Zeiler, M. D. (2012). Adadelta: an adaptive learning rate
method. arXiv preprint arXiv:1212.5701.

ADAMw: Loshchilov, I., & Hutter, F. (2017). Decoupled weight decay
regularization. arXiv preprint arXiv:1711.05101.

## See also

[`predict.brulee_mlp()`](https://brulee.tidymodels.org/dev/reference/predict.brulee_mlp.md),
[`coef.brulee_mlp()`](https://brulee.tidymodels.org/dev/reference/brulee-coefs.md),
[`autoplot.brulee_mlp()`](https://brulee.tidymodels.org/dev/reference/brulee-autoplot.md)

## Examples

``` r
# \donttest{
if (torch::torch_is_installed() & rlang::is_installed(c("recipes", "yardstick", "modeldata"))) {

 ## -----------------------------------------------------------------------------
 # regression examples (increase # epochs to get better results)

 data(ames, package = "modeldata")

 ames$Sale_Price <- log10(ames$Sale_Price)

 set.seed(122)
 in_train <- sample(1:nrow(ames), 2000)
 ames_train <- ames[ in_train,]
 ames_test  <- ames[-in_train,]


 # Using matrices
 set.seed(1)
 fit <-
   brulee_mlp(x = as.matrix(ames_train[, c("Longitude", "Latitude")]),
               y = ames_train$Sale_Price, penalty = 0.10)

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
 fit <- brulee_mlp(ames_rec, data = ames_train, hidden_units = 20,
                    dropout = 0.05, rate_schedule = "cyclic", step_size = 4)
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

 # Using multiple hidden layers and activation functions
 set.seed(2)
 hidden_fit <- brulee_mlp(ames_rec, data = ames_train,
                    hidden_units = c(15L, 17L), activation = c("relu", "elu"),
                    dropout = 0.05, rate_schedule = "cyclic", step_size = 4)

 predict(hidden_fit, ames_test) |>
   bind_cols(ames_test) |>
   rmse(Sale_Price, .pred)

 # ------------------------------------------------------------------------------
 # classification

 library(dplyr)
 library(ggplot2)

 data("parabolic", package = "modeldata")

 set.seed(1)
 in_train <- sample(1:nrow(parabolic), 300)
 parabolic_tr <- parabolic[ in_train,]
 parabolic_te <- parabolic[-in_train,]

 set.seed(2)
 cls_fit <- brulee_mlp(class ~ ., data = parabolic_tr, hidden_units = 2,
                        epochs = 200L, learn_rate = 0.1, activation = "elu",
                        penalty = 0.1, batch_size = 2^8, optimizer = "SGD")
 autoplot(cls_fit)

 grid_points <- seq(-4, 4, length.out = 100)

 grid <- expand.grid(X1 = grid_points, X2 = grid_points)

 predict(cls_fit, grid, type = "prob") |>
  bind_cols(grid) |>
  ggplot(aes(X1, X2)) +
  geom_contour(aes(z = .pred_Class1), breaks = 1/2, col = "black") +
  geom_point(data = parabolic_te, aes(col = class))

 }

# }
```
