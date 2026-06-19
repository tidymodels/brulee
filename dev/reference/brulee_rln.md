# Fit Regularization Learning Networks (RLN)

`brulee_rln()` fits a single-hidden-layer neural network where each
weight learns its own adaptive regularization coefficient.

## Usage

``` r
brulee_rln(x, ...)

# Default S3 method
brulee_rln(x, ...)

# S3 method for class 'data.frame'
brulee_rln(
  x,
  y,
  epochs = 100L,
  hidden_units = 5L,
  penalty_type = "L1",
  penalty_average = 1e-10,
  step_rate = 1e+06,
  activation = "relu",
  validation = 0.1,
  optimizer = "ADAMw",
  learn_rate = 0.001,
  rate_schedule = "none",
  momentum = 0,
  batch_size = NULL,
  stop_iter = 20,
  verbose = FALSE,
  device = NULL,
  ...
)

# S3 method for class 'matrix'
brulee_rln(
  x,
  y,
  epochs = 100L,
  hidden_units = 5L,
  penalty_type = "L1",
  penalty_average = 1e-10,
  step_rate = 1e+06,
  activation = "relu",
  validation = 0.1,
  optimizer = "ADAMw",
  learn_rate = 0.001,
  rate_schedule = "none",
  momentum = 0,
  batch_size = NULL,
  stop_iter = 20,
  verbose = FALSE,
  device = NULL,
  ...
)

# S3 method for class 'formula'
brulee_rln(
  formula,
  data,
  epochs = 100L,
  hidden_units = 5L,
  penalty_type = "L1",
  penalty_average = 1e-10,
  step_rate = 1e+06,
  activation = "relu",
  validation = 0.1,
  optimizer = "ADAMw",
  learn_rate = 0.001,
  rate_schedule = "none",
  momentum = 0,
  batch_size = NULL,
  stop_iter = 20,
  verbose = FALSE,
  device = NULL,
  ...
)

# S3 method for class 'recipe'
brulee_rln(
  x,
  data,
  epochs = 100L,
  hidden_units = 5L,
  penalty_type = "L1",
  penalty_average = 1e-10,
  step_rate = 1e+06,
  activation = "relu",
  validation = 0.1,
  optimizer = "ADAMw",
  learn_rate = 0.001,
  rate_schedule = "none",
  momentum = 0,
  batch_size = NULL,
  stop_iter = 20,
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

  - A **data frame** with 1 column (numeric or factor).

  - A **matrix** with numeric column (numeric or factor).

  - A **vector** (numeric or factor).

- epochs:

  An integer for the number of epochs of training.

- hidden_units:

  An integer for the number of units in the single hidden layer. Must be
  \>= 1.

- penalty_type:

  A string for the regularization norm: `"L1"` (default) or `"L2"`. L1
  is recommended by the original paper.

- penalty_average:

  A positive numeric value for the target geometric mean of the
  per-weight regularization coefficients (Theta in Shavitt and Segal
  (2018)), on the natural scale. Converted to log10 scale internally.
  Default is `1e-10` (i.e., `10^-10`).

- step_rate:

  A positive numeric value for the step size used to update the
  per-weight regularization coefficients (nu in Shavitt and Segal
  (2018)), on the natural scale. Converted to log10 scale internally;
  the multiplier applied is `10^log10(step_rate)`. Default is `1e6`
  (i.e., `10^6`). Both parameters are best tuned on the log10 scale.

- activation:

  A character vector for the activation function (such as "relu",
  "tanh", "sigmoid", and so on). See
  [`brulee_activations()`](https://brulee.tidymodels.org/dev/reference/brulee_activations.md)
  for a list of possible values. If `hidden_units` is a vector,
  `activation` can be a character vector with length equals to
  `length(hidden_units)` specifying the activation for each hidden
  layer.

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

A `brulee_rln` object with elements:

- `model_obj`: a serialized raw vector for the torch module.

- `estimates`: a list of model parameter matrices per epoch.

- `best_epoch`: an integer for the epoch with the smallest loss.

- `loss`: a numeric vector of loss values (scaled MSE) at each epoch.

- `dims`: a list of data dimensions.

- `y_stats`: a list of mean and standard deviation for the outcome.

- `parameters`: a list of tuning parameter values.

- `device`: a character string for the device used during training.

- `blueprint`: the `hardhat` blueprint data.

## Details

This function fits Regularization Learning Network (RLN) models for
regression (numeric outcomes only). Unlike standard regularization,
which applies a single global penalty, RLN learns a separate
regularization coefficient for each weight in the hidden layer. After
each gradient step, the per-weight coefficients (lambdas) are updated
and projected to keep their mean at `log10(penalty_average) * log(10)`.

### Why Use RLN?

RLNs are designed for tabular datasets where interpretability matters.
The per-weight regularization tends to produce very sparse networks. The
original paper reports eliminating up to ~99.8% of network edges and
~82% of input features. This sparsity makes it easier to identify which
inputs the network considers important, and the resulting models are
competitive with gradient boosted trees. The best results in the paper
are achieved by ensembling RLNs with gradient boosting tree ensembles.

### Architecture

The network is a single-hidden-layer MLP:

- Linear transformation (predictors -\> `hidden_units`)

- Activation function

- Linear transformation (`hidden_units` -\> 1 output)

Weights are initialized with Xavier normal initialization.

### RLN Update

After each optimizer step, the per-weight regularization coefficients
are updated using the gradient of the Counterfactual Loss with respect
to the coefficients, then projected onto a simplex so that
`mean(lambda) == log10(penalty_average) * log(10)`. The ADAMw optimizer
is the default.

### Other Notes

The outcome is internally standardized to have mean zero and standard
deviation one. Predictions are returned on the original scale.

By default, training halts when the validation loss increases for at
least `stop_iter` consecutive iterations. If `validation = 0` the
training set loss is used. The default for `stop_iter` is higher for RLN
than for other brulee models (20 vs 5) because the sparsification
process takes approximately 10-20 epochs to stabilize (Shavitt & Segal,
2018); stopping too early prevents the per-weight regularization from
taking effect.

Predictors should all be numeric and on comparable scales. Categorical
predictors must be converted to dummy variables.

Model parameters are saved each epoch so that `epoch` can be tuned
efficiently via the `epoch` argument of
[`predict.brulee_rln()`](https://brulee.tidymodels.org/dev/reference/predict.brulee_rln.md)
and
[`coef.brulee_rln()`](https://brulee.tidymodels.org/dev/reference/brulee-coefs.md).

## References

Shavitt, I., & Segal, E. (2018). Regularization learning networks: Deep
learning for tabular datasets. In *Advances in neural information
processing systems* (pp. 1379-1389).

## See also

[`predict.brulee_rln()`](https://brulee.tidymodels.org/dev/reference/predict.brulee_rln.md),
[`coef.brulee_rln()`](https://brulee.tidymodels.org/dev/reference/brulee-coefs.md),
[`autoplot.brulee_rln()`](https://brulee.tidymodels.org/dev/reference/brulee-autoplot.md)

## Examples

``` r
# \donttest{
if (torch::torch_is_installed() & rlang::is_installed(c("recipes", "yardstick", "modeldata"))) {

 data(ames, package = "modeldata")
 ames$Sale_Price <- log10(ames$Sale_Price)

 set.seed(122)
 in_train <- sample(1:nrow(ames), 2000)
 ames_train <- ames[ in_train,]
 ames_test  <- ames[-in_train,]

 library(recipes)

 ames_rec <-
  recipe(Sale_Price ~ Longitude + Latitude, data = ames_train) |>
    step_normalize(all_numeric_predictors())

 set.seed(2)
 fit <- brulee_rln(ames_rec, data = ames_train, hidden_units = 20L, epochs = 50L)
 fit

 autoplot(fit)

 library(yardstick)
 predict(fit, ames_test) |>
   bind_cols(ames_test) |>
   rmse(Sale_Price, .pred)

}
#> # A tibble: 1 × 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 rmse    standard       0.137
# }
```
