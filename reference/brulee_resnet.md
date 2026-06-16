# Fit residual neural networks (ResNet)

`brulee_resnet()` fits residual network models with skip connections.

## Usage

``` r
brulee_resnet(x, ...)

# Default S3 method
brulee_resnet(x, ...)

# S3 method for class 'data.frame'
brulee_resnet(
  x,
  y,
  epochs = 100L,
  hidden_units = 3L,
  bottleneck_units = hidden_units,
  residual_at = NULL,
  activation = "relu",
  penalty = 0.001,
  mixture = 0,
  dropout = 0,
  validation = 0.1,
  optimizer = "ADAMw",
  learn_rate = 0.01,
  rate_schedule = "none",
  momentum = 0,
  batch_size = NULL,
  class_weights = NULL,
  stop_iter = 5,
  grad_value_clip = 5,
  grad_norm_clip = 5,
  verbose = FALSE,
  device = NULL,
  ...
)

# S3 method for class 'matrix'
brulee_resnet(
  x,
  y,
  epochs = 100L,
  hidden_units = 3L,
  bottleneck_units = hidden_units,
  residual_at = NULL,
  activation = "relu",
  penalty = 0.001,
  mixture = 0,
  dropout = 0,
  validation = 0.1,
  optimizer = "ADAMw",
  learn_rate = 0.01,
  rate_schedule = "none",
  momentum = 0,
  batch_size = NULL,
  class_weights = NULL,
  stop_iter = 5,
  grad_value_clip = 5,
  grad_norm_clip = 5,
  verbose = FALSE,
  device = NULL,
  ...
)

# S3 method for class 'formula'
brulee_resnet(
  formula,
  data,
  epochs = 100L,
  hidden_units = 3L,
  bottleneck_units = hidden_units,
  residual_at = NULL,
  activation = "relu",
  penalty = 0.001,
  mixture = 0,
  dropout = 0,
  validation = 0.1,
  optimizer = "ADAMw",
  learn_rate = 0.01,
  rate_schedule = "none",
  momentum = 0,
  batch_size = NULL,
  class_weights = NULL,
  stop_iter = 5,
  grad_value_clip = 5,
  grad_norm_clip = 5,
  verbose = FALSE,
  device = NULL,
  ...
)

# S3 method for class 'recipe'
brulee_resnet(
  x,
  data,
  epochs = 100L,
  hidden_units = 3L,
  bottleneck_units = hidden_units,
  residual_at = NULL,
  activation = "relu",
  penalty = 0.001,
  mixture = 0,
  dropout = 0,
  validation = 0.1,
  optimizer = "ADAMw",
  learn_rate = 0.01,
  rate_schedule = "none",
  momentum = 0,
  batch_size = NULL,
  class_weights = NULL,
  stop_iter = 5,
  grad_value_clip = 5,
  grad_norm_clip = 5,
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

  - A **data frame** with 1 column (numeric or factor).

  - A **matrix** with numeric column (numeric or factor).

  - A **vector** (numeric or factor).

- epochs:

  An integer for the number of epochs of training.

- hidden_units:

  An integer vector specifying the number of hidden units in each layer.
  The length of this vector determines the number of layers. Each value
  must be \>= 1.

- bottleneck_units:

  An integer vector specifying the intermediate dimension within each
  layer. Must have the same length as `hidden_units`. Each value must be
  \>= 2.

- residual_at:

  An integer vector specifying which layer indices should have residual
  (skip) connections. For example, `residual_at = c(2, 4)` creates
  residual connections after layers 2 and 4, forming two residual blocks
  (layers 1-2 and 3-4). If `NULL` (default), every layer gets its own
  skip connection. Use `integer(0)` for no residual connections (i.e., a
  purely feed-forward model only).

- activation:

  A character vector for the activation function (such as "relu",
  "tanh", "sigmoid", and so on). See
  [`brulee_activations()`](https://brulee.tidymodels.org/reference/brulee_activations.md)
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
  [`schedule_decay_time()`](https://brulee.tidymodels.org/reference/schedule_decay_time.md)
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

A `brulee_resnet` object with elements:

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

This function fits residual network (ResNet) models for regression (when
the outcome is a number) or classification (a factor). ResNets use skip
connections that add the input of a block to its output, allowing
gradients to flow more easily through deep networks. For regression, the
mean squared error is optimized and cross-entropy is the loss function
for classification.

### Architecture

The network consists of a sequence of layers, each with batch
normalization, two linear transformations (with an intermediate
bottleneck dimension), and activation functions. Residual (skip)
connections can be placed at specified layers via the `residual_at`
parameter.

Each layer follows this pattern:

- Batch normalization (input dimension)

- Linear transformation (input dimension -\> `bottleneck_units[i]`)

- Activation function (ReLU by default)

- Dropout (if specified)

- Linear transformation (`bottleneck_units[i]` -\> `hidden_units[i]`)

- Dropout (if specified)

When a residual connection is specified at layer `i` via `residual_at`,
the output of layer `i` is added to the input from the start of that
residual block. If dimensions don't match, a linear projection is
automatically added.

### Residual Blocks

The `residual_at` parameter defines where skip connections occur:

- `residual_at = 3` creates one block spanning layers 1-3

- `residual_at = c(2, 4)` creates two blocks: layers 1-2 and layers 3-4

- `residual_at = NULL` (default) places a skip connection at every layer

- `residual_at = integer(0)` creates no residual connections (a purely
  feed-forward model)

### Learning Rates

The learning rate can be set to constant (the default) or dynamically
set via a learning rate scheduler (via the `rate_schedule`). Using
`rate_schedule = 'none'` uses the `learn_rate` argument. Otherwise, any
arguments to the schedulers can be passed via `...`.

### Other Notes

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

## References

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for
image recognition. In *Proceedings of the IEEE conference on computer
vision and pattern recognition* (pp. 770-778).

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Identity mappings in deep
residual networks. In *European conference on computer vision* (pp.
630-645). Springer, Cham.

Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021).
Revisiting deep learning models for tabular data. *Advances in neural
information processing systems*, 34, 18932-18943.

Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018).
Mobilenetv2: Inverted residuals and linear bottlenecks. In *Proceedings
of the IEEE conference on computer vision and pattern recognition* (pp.
4510-4520).

## See also

[`predict.brulee_resnet()`](https://brulee.tidymodels.org/reference/predict.brulee_resnet.md),
[`coef.brulee_resnet()`](https://brulee.tidymodels.org/reference/brulee-coefs.md),
[`autoplot.brulee_resnet()`](https://brulee.tidymodels.org/reference/brulee-autoplot.md)

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

 # Using recipe
 library(recipes)

 ames_rec <-
  recipe(Sale_Price ~ Longitude + Latitude, data = ames_train) |>
    step_normalize(all_numeric_predictors())

 set.seed(2)
 fit <- brulee_resnet(ames_rec, data = ames_train,
                      hidden_units = c(20, 10), bottleneck_units = c(15, 8),
                      residual_at = 2,
                      epochs = 50, batch_size = 32)
 fit

 summary(fit)

 autoplot(fit)

 library(yardstick)
 predict(fit, ames_test) |>
   bind_cols(ames_test) |>
   rmse(Sale_Price, .pred)

 # ------------------------------------------------------------------------------
 # classification

 library(dplyr)

 data("parabolic", package = "modeldata")

 set.seed(1)
 in_train <- sample(1:nrow(parabolic), 300)
 parabolic_tr <- parabolic[ in_train,]
 parabolic_te <- parabolic[-in_train,]

 set.seed(2)
 cls_fit <- brulee_resnet(class ~ ., data = parabolic_tr,
                          hidden_units = c(8, 5), bottleneck_units = c(6, 4),
                          residual_at = 1:2,
                          epochs = 200L, learn_rate = 0.1, activation = "elu",
                          penalty = 0.1, batch_size = 2^8)
 autoplot(cls_fit)

 predict(cls_fit, parabolic_te, type = "prob") |>
   bind_cols(parabolic_te) |>
   roc_auc(class, .pred_Class1)

 }
#> Residual network architecture
#> inputs: 2 | output dim: 1 | layers: 2
#> 
#> Residual group 1 (blocks 1-2, + skip)
#>   Block 1:
#>     BatchNorm1d(2)                        4 params
#>     Linear(2 -> 15)                      45 params
#>     ReLU                                  0 params
#>     Linear(15 -> 20)                    320 params
#>   Block 2:
#>     BatchNorm1d(20)                      40 params
#>     Linear(20 -> 8)                     168 params
#>     ReLU                                  0 params
#>     Linear(8 -> 10)                      90 params
#>   + skip: Linear(2 -> 10)                30 params
#> 
#> Output head
#>     BatchNorm1d(10)                      20 params
#>     Linear(10 -> 1)                      11 params
#> 
#> Total parameters: 728
#> # A tibble: 1 × 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 roc_auc binary         0.953
# }
```
