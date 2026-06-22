# Fit SAINT models for tabular data

`brulee_saint()` fits the SAINT (Self-Attention and Inter-sample
Attention Transformer) model from Somepalli *et al* (2021). SAINT
applies multi-head self-attention across both features (column
attention) and samples within a batch (row/inter-sample attention) to
learn complex feature interactions.

## Usage

``` r
brulee_saint(x, ...)

# Default S3 method
brulee_saint(x, ...)

# S3 method for class 'data.frame'
brulee_saint(
  x,
  y,
  epochs = 100L,
  num_embedding = 32L,
  attention_type = "both",
  num_attn_heads = 8L,
  num_attn_blocks = 6L,
  dropout_attn = 0.1,
  dropout_hidden = 0.1,
  dropout_last = 0,
  row_attention_on_predict = TRUE,
  hidden_units = 5,
  hidden_activations = "relu",
  penalty = 0.001,
  mixture = 0,
  validation = 0.1,
  optimizer = "ADAMw",
  learn_rate = 1e-04,
  rate_schedule = "none",
  momentum = 0,
  batch_size = NULL,
  class_weights = NULL,
  stop_iter = 5,
  grad_value_clip = 5,
  grad_norm_clip = 5,
  verbose = FALSE,
  device = NULL,
  target_token = TRUE,
  ...
)

# S3 method for class 'matrix'
brulee_saint(
  x,
  y,
  epochs = 100L,
  num_embedding = 32L,
  attention_type = "both",
  num_attn_heads = 8L,
  num_attn_blocks = 6L,
  dropout_attn = 0.1,
  dropout_hidden = 0.1,
  dropout_last = 0,
  row_attention_on_predict = TRUE,
  hidden_units = 5,
  hidden_activations = "relu",
  penalty = 0.001,
  mixture = 0,
  validation = 0.1,
  optimizer = "ADAMw",
  learn_rate = 1e-04,
  rate_schedule = "none",
  momentum = 0,
  batch_size = NULL,
  class_weights = NULL,
  stop_iter = 5,
  grad_value_clip = 5,
  grad_norm_clip = 5,
  verbose = FALSE,
  device = NULL,
  target_token = TRUE,
  ...
)

# S3 method for class 'formula'
brulee_saint(
  formula,
  data,
  epochs = 100L,
  num_embedding = 32L,
  attention_type = "both",
  num_attn_heads = 8L,
  num_attn_blocks = 6L,
  dropout_attn = 0.1,
  dropout_hidden = 0.1,
  dropout_last = 0,
  row_attention_on_predict = TRUE,
  hidden_units = 5,
  hidden_activations = "relu",
  penalty = 0.001,
  mixture = 0,
  validation = 0.1,
  optimizer = "ADAMw",
  learn_rate = 1e-04,
  rate_schedule = "none",
  momentum = 0,
  batch_size = NULL,
  class_weights = NULL,
  stop_iter = 5,
  grad_value_clip = 5,
  grad_norm_clip = 5,
  verbose = FALSE,
  device = NULL,
  target_token = TRUE,
  ...
)

# S3 method for class 'recipe'
brulee_saint(
  x,
  data,
  epochs = 100L,
  num_embedding = 32L,
  attention_type = "both",
  num_attn_heads = 8L,
  num_attn_blocks = 6L,
  dropout_attn = 0.1,
  dropout_hidden = 0.1,
  dropout_last = 0,
  row_attention_on_predict = TRUE,
  hidden_units = 5,
  hidden_activations = "relu",
  penalty = 0.001,
  mixture = 0,
  validation = 0.1,
  optimizer = "ADAMw",
  learn_rate = 1e-04,
  rate_schedule = "none",
  momentum = 0,
  batch_size = NULL,
  class_weights = NULL,
  stop_iter = 5,
  grad_value_clip = 5,
  grad_norm_clip = 5,
  verbose = FALSE,
  device = NULL,
  target_token = TRUE,
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

  An integer for the dimension of the initial embedding layer that
  encodes the original predictors. Each feature (categorical or
  continuous) is mapped to a vector of this dimension. Must be \>= 1.

- attention_type:

  A character string for the type of attention to use. Options are:

  - `"column"`: Column attention only (attends across features). This is
    the SAINT-s variant.

  - `"row"`: Row/inter-sample attention only (attends across samples
    within a batch). This is the SAINT-i variant.

  - `"both"`: Alternates between column and row attention in each
    transformer block. This is the full SAINT model.

- num_attn_heads:

  An integer for the number of parallel attention heads used in both
  column and row attention. Must be \>= 1.

- num_attn_blocks:

  An integer for the number of sequential transformer blocks (depth).
  Must be \>= 1.

- dropout_attn:

  A number in `[0, 1)` for the dropout rate applied to attention weights
  during training.

- dropout_hidden:

  A number in `[0, 1)` for the dropout rate applied within the
  feed-forward layers of each transformer block.

- dropout_last:

  A number in `[0, 1)` for the dropout rate applied between the last
  hidden layer and the output head. Only has effect when `hidden_units`
  is not `NULL`. Default is 0 (no dropout).

- row_attention_on_predict:

  A logical value. Should row (inter-sample) attention be applied during
  prediction? Default is `TRUE`, matching the training-time behavior.
  When `FALSE`, row attention is bypassed at predict time so that
  predictions for a given row do not depend on what other rows are in
  the prediction set; column attention is used on its own. This is only
  relevant when `attention_type` is `"row"` or `"both"`.

- hidden_units:

  An integer vector for the number of units in optional hidden layers
  between the transformer backbone and the output head. When `NULL` (the
  default), no hidden layers are added and the pooled transformer output
  is projected directly to the output.

- hidden_activations:

  A character vector of activation functions for the hidden layers. Must
  be the same length as `hidden_units` or a single value that will be
  recycled. See
  [`brulee_activations()`](https://brulee.tidymodels.org/dev/reference/brulee_activations.md)
  for options.

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

- device:

  A single character string for the device to train on (e.g., `"cpu"` or
  `"cuda"` for GPU). If `NULL`, the function will use the GPU if
  available, otherwise CPU. See
  [training_efficiency](https://brulee.tidymodels.org/dev/reference/training_efficiency.md).

- target_token:

  A logical value. When `TRUE` (the default), a learnable target token
  (`[CLS]` in the SAINT paper) is prepended to each sample's feature
  sequence and only its final-layer embedding is fed to the head. This
  matches the architecture described in the SAINT paper (Section 3 and
  Figure 1); see the **Target Token Pooling** section in **Details**.
  When `FALSE`, the head instead consumes the concatenation of every
  feature token, which matches the SAINT reference implementation at
  <https://github.com/somepago/saint>.

- formula:

  A formula specifying the outcome term(s) on the left-hand side, and
  the predictor term(s) on the right-hand side.

- data:

  When a **recipe** or **formula** is used, `data` is specified as:

  - A **data frame** containing both the predictors and the outcome.

## Value

A `brulee_saint` object with elements:

- `models_obj`: a serialized raw vector for the torch module.

- `estimates`: a list of matrices with the model parameter estimates per
  epoch.

- `best_epoch`: an integer for the epoch with the smallest loss.

- `loss`: A vector of loss values (MSE for regression, negative log-
  likelihood for classification) at each epoch.

- `dim`: A list of data dimensions and feature metadata.

- `y_stats`: A list of summary statistics for numeric outcomes.

- `parameters`: A list of some tuning parameter values.

- `device`: A character string for the device used during training.

- `blueprint`: The `hardhat` blueprint data.

## Details

### Architecture

The SAINT architecture has three stages:

1.  **Embedding layer**: Categorical features are mapped through
    per-feature embedding tables. Continuous features are passed through
    per-feature MLPs (1 -\> 100 -\> `num_embedding`). These initial
    embeddings are per-feature; there is a distinct embedding MLP for
    each predictor. Also, see the "Target Token Pooling" section below.

2.  **Transformer backbone**: A stack of `num_attn_blocks` transformer
    layers. Each layer contains multi-head self-attention followed by a
    feed-forward network with GeGLU activation. For
    `attention_type = "both"`, each block alternates between column
    attention (across features) and row attention (across samples within
    the batch).

3.  **Output head**: Pools the transformer output (either the target
    token's embedding or the flattened concatenation of all feature
    embeddings, controlled by `target_token`) and projects it through
    optional hidden layers to the output dimension.

There is a [`summary()`](https://rdrr.io/r/base/summary.html) methods
that can provide details of the architecture for a specific model fit.

Differences in this implementation and the original paper: pretraining
isn't supported.

### Attention Types

- **Column attention** (`"column"`): Standard self-attention over
  features. Each feature embedding attends to all other feature
  embeddings.

- **Row attention** (`"row"`): inter-sample attention. Reshapes the
  batch so that each sample's full feature representation becomes a
  single token, then applies attention across all samples in the batch.

- **Both** (`"both"`): Alternates between column and row attention in
  each transformer block. This is the full SAINT model.

### Target Token Pooling

Borrowing from BERT, SAINT prepends a learnable target token (the paper
calls it `[CLS]`) to each sample's feature sequence before the
transformer. With embeddings `E(x_i^{(1)}), ..., E(x_i^{(n)})` for the
`n` predictors of sample `i`, the input sequence becomes

`[target, E(x_i^{(1)}), E(x_i^{(2)}), ..., E(x_i^{(n)})]`

giving `n + 1` tokens of dimension `num_embedding`. The target token has
no input value; it is a free parameter of the model that is trained
alongside the rest of the network. Column attention lets every feature
token attend to the target and vice versa, so the target slot
accumulates a contextual summary of the sample. When `attention_type` is
`"row"` or `"both"`, inter-sample attention sees the full `n + 1` token
sequence per sample, so the target slot also exchanges information
across samples in the batch.

After the transformer backbone, the head reads *only* the final-layer
embedding of the target token (the first position) and feeds it through
the optional `hidden_units` MLP and the output layer. This is what the
paper describes in Figure 1: "We take the contextual embeddings from
SAINT and pass only the embedding correspond to the CLS token through an
MLP to obtain the final prediction."

With `target_token = FALSE`, no target token is added and the head
instead consumes the concatenation of all `n` feature tokens. That
option is provided because the SAINT reference Python implementation
(<https://github.com/somepago/saint>) departs from the paper and uses
flatten-pooling; it is kept available for compatibility with that code
path and for users who want the original brulee behavior.

### Row Attention at Prediction Time

Row attention computations adjust the internal embeddings based on the
rows that are available at any given time. During training, the other
rows in the batch are used to compute attention. After training, when
[`predict()`](https://rdrr.io/r/stats/predict.html) is called, the
default behavior is to keep row attention on, mirroring the
training-time computation. Because row attention is computed across the
samples present in a given call, predictions for a row depend on what
other rows are passed alongside it. To get batch-independent predictions
(where the prediction for a given row is the same regardless of what
other rows are in the input), set `row_attention_on_predict` to `FALSE`;
row attention is then bypassed at predict time and column attention is
used on its own.

### Learning Rates

The learning rate can be set to constant (the default) or dynamically
set via a learning rate scheduler (via the `rate_schedule`). Using
`rate_schedule = 'none'` uses the `learn_rate` argument.

### Other Notes

Unlike other brulee models, `brulee_saint()` natively handles factor
predictors via learned embeddings. Factor columns are automatically
detected and embedded, while numeric columns pass through per-feature
MLPs. There is *no need to pre-encode factors as indicators*.

When the outcome is a number, the function internally standardizes the
outcome data to have mean zero and a standard deviation of one. The
prediction function creates predictions on the original scale.

By default, training halts when the validation loss increases for at
least `stop_iter` iterations. If `validation = 0` the training set loss
is used.

The model objects are saved for each epoch so that the number of epochs
can be efficiently tuned. The
[`predict()`](https://rdrr.io/r/stats/predict.html) method for this
model has an `epoch` argument (which defaults to the epoch with the best
loss value).

## References

Somepalli, G., Goldblum, M., Schwarzschild, A., Bruss, C. B., &
Goldstein, T. (2021). SAINT: Improved Neural Networks for Tabular Data
via Row Attention and Contrastive Pre-Training. arXiv preprint
arXiv:2106.01342.

## See also

[`predict.brulee_saint()`](https://brulee.tidymodels.org/dev/reference/predict.brulee_saint.md),
[`autoplot.brulee_saint()`](https://brulee.tidymodels.org/dev/reference/brulee-autoplot.md)

## Examples

``` r
# \donttest{
pkgs <- c("recipes", "yardstick", "modeldata")
if (torch::torch_is_installed() & rlang::is_installed(pkgs)) {

 set.seed(87261)
 tr_data <- modeldata::sim_regression(500, method = "worley_1987")
 te_data <- modeldata::sim_regression(50, method = "worley_1987")

 library(recipes)
 rec <- recipe(outcome ~ ., data = te_data) |>
  step_normalize(all_numeric_predictors())

 set.seed(389)
 fit <- brulee_saint(
  rec,
  data = te_data,
  hidden_unit = 5,
  dropout_hidden = 0.2,
  num_embedding = 3,
  num_attn_heads = 5,
  num_attn_blocks = 4,
  dropout_attn = 0.2,
  epochs = 50L,
  batch_size = 32L,
  learn_rate = 0.01,
  optimize = "SGD",
  verbose = TRUE
 )

 autoplot(fit)
 summary(fit)

 library(yardstick)
 predict(fit, te_data) |>
  dplyr::bind_cols(te_data) |>
  rsq(outcome, .pred)

}
#> epoch: 00, learn rate: 0.01000, Loss (scaled): 0.311
#> epoch: 01, learn rate: 0.01000, Loss (scaled): 0.292
#> epoch: 02, learn rate: 0.01000, Loss (scaled): 0.288
#> epoch: 03, learn rate: 0.01000, Loss (scaled): 0.291
#> epoch: 04, learn rate: 0.01000, Loss (scaled): 0.301
#> epoch: 05, learn rate: 0.01000, Loss (scaled): 0.315
#> epoch: 06, learn rate: 0.01000, Loss (scaled): 0.296
#> epoch: 07, learn rate: 0.01000, Loss (scaled): 0.304
#> SAINT architecture
#> inputs: 8 (0 categorical, 8 numeric) | output dim: 1
#> attention: both | embedding dim: 3 | target token: TRUE
#> 
#> Embedding layer
#>   Target token (1 x 3)                         3 params
#>   8 x MLP(1 -> 100 -> 3)                   4,024 params
#> 
#> Transformer backbone (4 blocks, column + row attention)
#>   Block 1:
#>     Column attention:
#>       LayerNorm(3)                                 6 params
#>       Attention(dim=3, heads=5)                  963 params
#>       LayerNorm(3)                                 6 params
#>       FeedForward(3, GEGLU)                      135 params
#>     Row attention:
#>       LayerNorm(27)                               54 params
#>       Attention(dim=27, heads=5)              34,587 params
#>       LayerNorm(27)                               54 params
#>       FeedForward(27, GEGLU)                   8,991 params
#>   Block 2:
#>     Column attention:
#>       LayerNorm(3)                                 6 params
#>       Attention(dim=3, heads=5)                  963 params
#>       LayerNorm(3)                                 6 params
#>       FeedForward(3, GEGLU)                      135 params
#>     Row attention:
#>       LayerNorm(27)                               54 params
#>       Attention(dim=27, heads=5)              34,587 params
#>       LayerNorm(27)                               54 params
#>       FeedForward(27, GEGLU)                   8,991 params
#>   Block 3:
#>     Column attention:
#>       LayerNorm(3)                                 6 params
#>       Attention(dim=3, heads=5)                  963 params
#>       LayerNorm(3)                                 6 params
#>       FeedForward(3, GEGLU)                      135 params
#>     Row attention:
#>       LayerNorm(27)                               54 params
#>       Attention(dim=27, heads=5)              34,587 params
#>       LayerNorm(27)                               54 params
#>       FeedForward(27, GEGLU)                   8,991 params
#>   Block 4:
#>     Column attention:
#>       LayerNorm(3)                                 6 params
#>       Attention(dim=3, heads=5)                  963 params
#>       LayerNorm(3)                                 6 params
#>       FeedForward(3, GEGLU)                      135 params
#>     Row attention:
#>       LayerNorm(27)                               54 params
#>       Attention(dim=27, heads=5)              34,587 params
#>       LayerNorm(27)                               54 params
#>       FeedForward(27, GEGLU)                   8,991 params
#> 
#> Hidden layers
#>   Linear(3 -> 5)                              20 params
#>   ReLU                                         0 params
#> 
#> Output head
#>   Linear(5 -> 1)                               6 params
#> 
#> Total parameters: 183,237
#> # A tibble: 1 × 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 rsq     standard       0.160
# }
```
