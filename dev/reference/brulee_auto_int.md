# Fit AutoInt models for tabular data

`brulee_auto_int()` fits AutoInt from Song *at al* (2019) that use
multi-head columnar self-attention to help exploit how combinations of
embeddings can be used to improve specific predictions.

## Usage

``` r
brulee_auto_int(x, ...)

# Default S3 method
brulee_auto_int(x, ...)

# S3 method for class 'data.frame'
brulee_auto_int(
  x,
  y,
  epochs = 100L,
  num_embedding = 16L,
  num_attn_feat = 16L,
  num_attn_heads = 2L,
  num_attn_blocks = 3L,
  activation = "relu",
  hidden_units = NULL,
  hidden_activations = NULL,
  penalty = 0.001,
  mixture = 0,
  dropout = 0,
  dropout_attn = 0,
  dropout_embedding = 0,
  validation = 0.1,
  optimizer = "ADAMw",
  learn_rate = 0.01,
  rate_schedule = "none",
  momentum = 0,
  batch_size = NULL,
  class_weights = NULL,
  stop_iter = 5,
  verbose = FALSE,
  device = NULL,
  ...
)

# S3 method for class 'matrix'
brulee_auto_int(
  x,
  y,
  epochs = 100L,
  num_embedding = 16L,
  num_attn_feat = 16L,
  num_attn_heads = 2L,
  num_attn_blocks = 3L,
  activation = "relu",
  hidden_units = NULL,
  hidden_activations = NULL,
  dropout = 0,
  penalty = 0.001,
  mixture = 0,
  dropout_attn = 0,
  dropout_embedding = 0,
  validation = 0.1,
  optimizer = "ADAMw",
  learn_rate = 0.01,
  rate_schedule = "none",
  momentum = 0,
  batch_size = NULL,
  class_weights = NULL,
  stop_iter = 5,
  verbose = FALSE,
  device = NULL,
  ...
)

# S3 method for class 'formula'
brulee_auto_int(
  formula,
  data,
  epochs = 100L,
  num_embedding = 16L,
  num_attn_feat = 16L,
  num_attn_heads = 2L,
  num_attn_blocks = 3L,
  activation = "relu",
  hidden_units = NULL,
  hidden_activations = NULL,
  dropout = 0,
  penalty = 0.001,
  mixture = 0,
  dropout_attn = 0,
  dropout_embedding = 0,
  validation = 0.1,
  optimizer = "ADAMw",
  learn_rate = 0.01,
  rate_schedule = "none",
  momentum = 0,
  batch_size = NULL,
  class_weights = NULL,
  stop_iter = 5,
  verbose = FALSE,
  device = NULL,
  ...
)

# S3 method for class 'recipe'
brulee_auto_int(
  x,
  data,
  epochs = 100L,
  num_embedding = 16L,
  num_attn_feat = 16L,
  num_attn_heads = 2L,
  num_attn_blocks = 3L,
  activation = "relu",
  hidden_units = NULL,
  hidden_activations = NULL,
  dropout = 0,
  penalty = 0.001,
  mixture = 0,
  dropout_attn = 0,
  dropout_embedding = 0,
  validation = 0.1,
  optimizer = "ADAMw",
  learn_rate = 0.01,
  rate_schedule = "none",
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

  - A **data frame** with 1 column (numeric or factor).

  - A **matrix** with numeric column (numeric or factor).

  - A **vector** (numeric or factor).

- epochs:

  An integer for the number of epochs of training.

- num_embedding:

  An integer for the embedding dimension. Each feature (categorical or
  continuous) is mapped to a vector of this dimension. Must be \>= 1.

- num_attn_feat:

  An integer for the per-head attention dimension. The total attention
  dimension is `num_attn_feat * num_attn_heads`. Must be \>= 1.

- num_attn_heads:

  An integer for the number of attention heads. Each head learns
  different interaction patterns in parallel. Must be \>= 1.

- num_attn_blocks:

  An integer for the number of stacked self-attention layers. More
  layers capture higher-order interactions. Must be \>= 1.

- activation:

  A single character string for the activation function used in the
  self-attention backbone (applied after each residual connection in
  each attention block). This does not affect the optional hidden
  layers; use `hidden_activations` for those. See
  [`brulee_activations()`](https://brulee.tidymodels.org/dev/reference/brulee_activations.md)
  for options.

- hidden_units:

  An integer vector for the number of units in optional hidden layers
  between the attention backbone and the output head. For example,
  `c(64L, 32L)` adds two hidden layers with 64 and 32 units. When `NULL`
  (the default), no hidden layers are added.

- hidden_activations:

  A character vector of activation functions for the hidden layers. Must
  be the same length as `hidden_units` or a single value that will be
  recycled. When `NULL` (the default), no hidden layers are added. See
  [`brulee_activations()`](https://brulee.tidymodels.org/dev/reference/brulee_activations.md)
  for options.

- penalty:

  The amount of weight decay (i.e., L2 regularization).

- mixture:

  Proportion of Lasso Penalty (type: double, default: 0.0). A value of
  mixture = 1 corresponds to a pure lasso model, while mixture = 0
  indicates ridge regression (a.k.a weight decay). Must be zero for
  optimizers `"ADAMw"`, `"RMSprop"`, `"Adadelta"`.

- dropout:

  A number in `[0, 1)` for the dropout rate applied between the last
  hidden layer and the output head. Only has effect when `hidden_units`
  is not `NULL`. Default is 0 (no dropout).

- dropout_attn:

  A number in `[0, 1)` for the dropout rate applied to attention weights
  during training.

- dropout_embedding:

  A number in `[0, 1)` for the dropout rate applied to the embedding
  layer during training.

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

A `brulee_auto_int` object with elements:

- `models_obj`: a serialized raw vector for the torch module.

- `estimates`: a list of matrices with the model parameter estimates per
  epoch.

- `best_epoch`: an integer for the epoch with the smallest loss.

- `loss`: A vector of loss values (MSE for regression, negative log-
  likelihood for classification) at each epoch.

- `dim`: A list of data dimensions and feature metadata.

- `top_interactions`: A tibble containing the top 10 two-way feature
  interactions.

- `y_stats`: A list of summary statistics for numeric outcomes.

- `parameters`: A list of some tuning parameter values.

- `device`: A character string for the device used during training.

- `blueprint`: The `hardhat` blueprint data.

## Details

### What is Being Estimated

In statistics, an interaction occurs when two or more predictors jointly
predict the outcome. You need to know the values of all predictors
within the interaction effect to appropriately model the data. AutoInt
is often described as "automatically learning feature interactions," but
that is not an accurate description.

In neural networks, the original predictors are converted to
*embeddings*, which are often the hidden units of the network.

AutoInt uses *column attention* to change how embeddings are
represented. It learns how to make the embeddings more relevant to the
outcome by creating mixtures of them. For example, if we predict a data
point in one part of the predictor space, attention will refocus (i.e.,
transform) the embedding to be more relevant to that part of the space.

### Architecture

The AutoInt architecture has three stages:

1.  **Embedding layer**: Maps every feature (categorical or continuous)
    into a shared vector space of dimension `num_embedding`.

2.  **Self-attention backbone**: A stack of `num_attn_blocks` multi-head
    self-attention layers. After all blocks, a residual connection from
    the original embeddings is added and an activation is applied.

3.  **Hidden layers** (optional): If `hidden_units` is specified, one or
    more fully-connected layers with activations process the flattened
    attention output before the output head.

4.  **Output head**: Projects to the output dimension via a linear
    layer.

Unlike other brulee models, `brulee_auto_int()` natively handles factor
predictors via learned embeddings. Factor columns are automatically
detected and embedded, while numeric columns use a scaled embedding.
There is *no need to pre-encode factors as indicators*.

### Attention Parameters

The self-attention backbone has several tuning parameters that control
its capacity and regularization:

- `num_attn_heads`: The number of attention heads that operate **in
  parallel** within each attention block. Each head independently learns
  which features interact, giving the model multiple "views" of the
  feature relationships. The total attention dimension per block is
  `num_attn_feat * num_attn_heads`.

- `num_attn_feat`: The per-head attention dimension. Each head projects
  features into a space of this size to compute attention scores. Larger
  values give each head more capacity to represent complex interactions.

- `num_attn_blocks`: The number of attention layers stacked
  **sequentially**. Each block's output feeds into the next, allowing
  the model to build higher-order interactions (e.g., block 1 captures
  pairwise interactions, block 2 can combine those into three-way
  interactions, etc.).

- `activation`: The activation function applied after the residual
  connection at the end of the attention backbone.

- `dropout_attn`: Dropout applied to the attention weight matrix within
  each block, which randomly zeroes out attention connections during
  training.

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
least `stop_iter` iterations. If `validation = 0` the training set loss
is used.

The model objects are saved for each epoch so that the number of epochs
can be efficiently tuned. Both the
[`predict()`](https://rdrr.io/r/stats/predict.html) method for this
model has an `epoch` argument (which defaults to the epoch with the best
loss value).

The use of the L1 penalty (a.k.a. the lasso penalty) does *not* force
parameters to be strictly zero (as it does in packages such as glmnet).
The zeroing out of parameters is a specific feature the optimization
method used in those packages.

## References

Song, W., Shi, C., Xiao, Z., Duan, Z., Xu, Y., Zhang, M., & Tang, J.
(2019). AutoInt: Automatic Feature Interaction Learning via
Self-Attentive Neural Networks. In *Proceedings of the 28th ACM
International Conference on Information and Knowledge Management
(CIKM)*.

## See also

[`predict.brulee_auto_int()`](https://brulee.tidymodels.org/dev/reference/predict.brulee_auto_int.md),
[`autoplot.brulee_auto_int()`](https://brulee.tidymodels.org/dev/reference/brulee-autoplot.md)

## Examples

``` r
# \donttest{
pkgs <- c("recipes", "yardstick", "modeldata")
if (torch::torch_is_installed() & rlang::is_installed(pkgs)) {

  set.seed(87261)
  tr_data <- modeldata::sim_regression(500)
  te_data <- modeldata::sim_regression(50)

  set.seed(2)
  fit <- brulee_auto_int(outcome ~ ., data = tr_data,
                         epochs = 50L, batch_size = 64L, stop_iter = 10L,
                         learn_rate = 0.01, penalty = 0.01)
  fit

  autoplot(fit)

  library(yardstick)
  predict(fit, te_data) |>
   dplyr::bind_cols(te_data) |>
   rmse(outcome, .pred)

}
#> # A tibble: 1 × 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 rmse    standard        16.5
# }
```
