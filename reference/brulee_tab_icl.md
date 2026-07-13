# Fit a TabICL tabular foundation model

`brulee_tab_icl()` prepares the pre-trained TabICL (Tabular In-Context
Learning) foundation model from Qu *et al* (2025) for prediction. TabICL
is a transformer that makes predictions on tabular data by *in-context
learning*: it is not trained on your data. Instead, the released
pre-trained weights are loaded and the model conditions on your training
rows at prediction time, much like a few-shot language model conditions
on its prompt. Both classification and regression are supported.

## Usage

``` r
brulee_tab_icl(x, ...)

# Default S3 method
brulee_tab_icl(x, ...)

# S3 method for class 'data.frame'
brulee_tab_icl(
  x,
  y,
  num_estimators = 8L,
  normalization = c("none", "YeoJohnson"),
  softmax_temperature = 0.9,
  training_set_limit = Inf,
  device = NULL,
  ...
)

# S3 method for class 'matrix'
brulee_tab_icl(
  x,
  y,
  num_estimators = 8L,
  normalization = c("none", "YeoJohnson"),
  softmax_temperature = 0.9,
  training_set_limit = Inf,
  device = NULL,
  ...
)

# S3 method for class 'formula'
brulee_tab_icl(
  formula,
  data,
  num_estimators = 8L,
  normalization = c("none", "YeoJohnson"),
  softmax_temperature = 0.9,
  training_set_limit = Inf,
  device = NULL,
  ...
)

# S3 method for class 'recipe'
brulee_tab_icl(
  x,
  data,
  num_estimators = 8L,
  normalization = c("none", "YeoJohnson"),
  softmax_temperature = 0.9,
  training_set_limit = Inf,
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

  Not currently used.

- y:

  When `x` is a **data frame** or **matrix**, `y` is the outcome
  specified as:

  - A **data frame** with 1 column (numeric or factor).

  - A **matrix** with numeric column (numeric or factor).

  - A **vector** (numeric or factor).

- num_estimators:

  An integer for the number of ensemble members (default `8`). Each
  member preprocesses, permutes features, and (for classification)
  shuffles class labels differently; their predictions are averaged. Use
  `1` for a single, fully deterministic member.

- normalization:

  A character vector of per-member normalization methods. Currently
  `"none"` (standardize only) and `"YeoJohnson"` (Yeo-Johnson power
  transform on top of standardization) are supported.

- softmax_temperature:

  A number for the temperature applied to the classification softmax.
  Only used for classification.

- training_set_limit:

  A single number giving the maximum number of training rows kept as
  in-context examples. When the training data has more rows than this, a
  subsample of exactly `training_set_limit` rows is drawn (stratified by
  the outcome for classification, simple random for regression). The
  default is `Inf`, which keeps every row. Useful for capping memory and
  prediction time on large training sets, since the entire (kept)
  training set is stored on the fitted object and re-sent through the
  network on every call to
  [`predict()`](https://rdrr.io/r/stats/predict.html).

- device:

  A character string for the compute device: `"cpu"` (the default) or
  `"cuda"`. See the **Device support** section.

- formula:

  A formula specifying the outcome term(s) on the left-hand side, and
  the predictor term(s) on the right-hand side.

- data:

  When a **recipe** or **formula** is used, `data` is specified as:

  - A **data frame** containing both the predictors and the outcome.

## Value

A `brulee_tab_icl` object with elements:

- `path`: the cached checkpoint directory the weights are loaded from.

- `config`: the parsed model configuration.

- `task`: `"classification"` or `"regression"`.

- `levels`: the outcome factor levels (classification only).

- `encoders`: the fitted per-column predictor encoders.

- `train_x`, `train_y`: the encoded training context.

- `num_estimators`, `normalization`, `softmax_temperature`: ensemble
  settings.

- `device`: the resolved compute device.

- `blueprint`: the `hardhat` blueprint.

## Details

### In-context learning

Unlike the other brulee models, `brulee_tab_icl()` does not train any
parameters. The pre-trained network is fixed; "fitting" simply validates
and stores the (encoded) training predictors and outcomes together with
a reference to the checkpoint. At
[`predict()`](https://rdrr.io/r/stats/predict.html) time, the model is
given the training rows as labelled context alongside the new rows and
produces predictions in a single forward pass. Because the training data
are stored on the fitted object, larger training sets make the object
larger and prediction slower.

### Architecture

TabICL processes a table through three transformer stages:

1.  **Column embedding**: a per-column set transformer turns each cell
    into a distribution-aware embedding, optionally informed by the
    target.

2.  **Row interaction**: a transformer with rotary position encoding
    mixes the feature embeddings within each row and aggregates them
    with learnable CLS tokens.

3.  **In-context learning**: a dataset-level transformer lets the test
    rows attend to the labelled training rows to produce class logits
    (classification) or quantiles (regression).

### Preprocessing

TabICL applies its own preprocessing to mirror the reference
implementation, so most data shaping that other tabular models require
is unnecessary (and in some cases counter-productive). The pipeline runs
in two stages.

**Stage 1: numeric encoding (always, at fit time).**

Each predictor column is converted to a numeric value:

- Factor and character columns are **ordinal-encoded**: the unique
  values seen during fitting are sorted lexicographically and mapped to
  0-based integers. *Do not pre-encode factors as indicator (dummy)
  variables.* TabICL is a per-column tokenized transformer; one ordinal
  column gives the model one token per row, while a wide one-hot
  expansion bloats the sequence length, blows up the row-interaction
  stage, and degrades prediction quality.

- Numeric columns are taken as-is.

The training predictors are stored on the fitted object in this encoded
form so that they can serve as context at
[`predict()`](https://rdrr.io/r/stats/predict.html) time.

**Stage 2: per-member normalization (at predict time).**

For each ensemble member, the encoded predictors pass through a small
pipeline before being handed to the network:

1.  **Standardization** — center by column mean and divide by the
    population standard deviation (with a small epsilon and a soft clip
    to \\\pm 100\\). This always runs.

2.  **Optional Yeo-Johnson** — when the member's `normalization` slot is
    `"YeoJohnson"`, a per-column Yeo-Johnson power transform is inserted
    between standardization and outlier clipping. The Yeo-Johnson
    \\\lambda\\ for each column is fit on the standardized training data
    via maximum likelihood, then the transformed values are
    re-standardized so the downstream stages see the same mean/scale as
    the `"none"` path. The transform is helpful when individual columns
    are heavily skewed or heavy-tailed. The `normalization` argument is
    a vector because the default ensemble intentionally mixes `"none"`
    and `"YeoJohnson"` across members to boost predictive diversity.

3.  **Outlier clipping** — a two-stage z-score clipper trims extreme
    values. This always runs.

All parameters in stage 2 (means, standard deviations, Yeo-Johnson
lambdas, clip bounds) are estimated on the training rows alone and then
applied to both training and new rows.

For regression, the outcome is standardized internally and predictions
are returned on the original scale. For classification, the outcome is
label-encoded.

### Missing Values

Missing values do not need to be imputed by the user.

- **Numeric columns**: at fit time the column mean (ignoring `NA`) is
  learned and reused to fill any `NA` in both the training context and
  the prediction rows.

- **Factor and character columns**: missing values, as well as any *new*
  factor levels seen at prediction time that were not present during
  fitting, are mapped to the sentinel code `-1` and treated as a
  distinct "unknown" category by the model.

Pre-imputation by the user is still allowed and is sometimes desirable
(for example, when a domain-appropriate imputation outperforms a column
mean), but it is not required for the model to run.

### Ensembling

With `num_estimators > 1`, several views of the data are created by
permuting features and (for classification) shuffling class labels, each
optionally with a different normalization. Class logits are averaged
across members and converted to probabilities with a temperature
softmax; regression means are averaged. `num_estimators = 1` uses a
single deterministic member (no shuffles, `"none"` normalization). Note
that with more than one member the feature permutations are drawn with
R's random number generator, so results are a faithful reproduction of
the reference ensemble but not bit-for-bit identical to it; set the seed
for reproducibility across runs.

### Device support

Computation runs on CPU by default and on CUDA when `device = "cuda"`
and a GPU is available. The Apple `"mps"` backend is **not** supported:
the bundled libtorch MPS kernels crash on parts of the model, so
requesting `"mps"` issues a warning and falls back to CPU.

### Pre-trained weights

The estimated parameters from the pre-trained Python model are used.
These weights (more than 200MB) are not shipped with the package and are
cached once with
[`tab_icl_download_weights()`](https://brulee.tidymodels.org/reference/tab_icl_download_weights.md).
If they are not cached when `brulee_tab_icl()` runs, it prompts to
download them in an interactive session and errors (pointing you to
[`tab_icl_download_weights()`](https://brulee.tidymodels.org/reference/tab_icl_download_weights.md))
otherwise.

## References

Qu, J., Holzmüller, D., Varoquaux, G., & Le Morvan, M. (2025). TabICL: A
Tabular Foundation Model for In-Context Learning on Large Data. arXiv
preprint arXiv:2502.05564.

## See also

[`predict.brulee_tab_icl()`](https://brulee.tidymodels.org/reference/predict.brulee_tab_icl.md)

## Examples

``` r
if (FALSE) { # \dontrun{
# Requires cached TabICL weights; download them first with
# tab_icl_download_weights() (see the "Pre-trained weights" section).

if (torch::torch_is_installed() && rlang::is_installed("modeldata")) {
  data(penguins, package = "modeldata")
  penguins <- na.omit(penguins)

  in_train <- sample(seq_len(nrow(penguins)), 250)
  tr <- penguins[in_train, ]
  te <- penguins[-in_train, ]

  # Classification (uses the cached classification checkpoint)
  cls_fit <- brulee_tab_icl(species ~ ., data = tr)
  predict(cls_fit, te)
  predict(cls_fit, te, type = "prob")

  # Regression (uses the cached regression checkpoint)
  reg_fit <- brulee_tab_icl(body_mass_g ~ ., data = tr)
  predict(reg_fit, te)
}
} # }
```
