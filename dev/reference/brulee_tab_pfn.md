# Fit a TabPFN model

`brulee_tab_pfn()` applies data to the pre-trained TabPFN tabular
foundation model of Hollmann *et al* (2025), which emulates Bayesian
inference for regression and classification. The model runs in R torch;
no Python is needed. The arguments mirror `tab_pfn()` in the tabpfn
package, which runs the Python implementation.

## Usage

``` r
brulee_tab_pfn(x, ...)

# Default S3 method
brulee_tab_pfn(x, ...)

# S3 method for class 'data.frame'
brulee_tab_pfn(
  x,
  y,
  num_estimators = NULL,
  softmax_temperature = NULL,
  balance_probabilities = FALSE,
  average_before_softmax = FALSE,
  training_set_limit = Inf,
  version = NULL,
  device = NULL,
  ignore_pretraining_limits = FALSE,
  ...
)

# S3 method for class 'matrix'
brulee_tab_pfn(
  x,
  y,
  num_estimators = NULL,
  softmax_temperature = NULL,
  balance_probabilities = FALSE,
  average_before_softmax = FALSE,
  training_set_limit = Inf,
  version = NULL,
  device = NULL,
  ignore_pretraining_limits = FALSE,
  ...
)

# S3 method for class 'formula'
brulee_tab_pfn(
  formula,
  data,
  num_estimators = NULL,
  softmax_temperature = NULL,
  balance_probabilities = FALSE,
  average_before_softmax = FALSE,
  training_set_limit = Inf,
  version = NULL,
  device = NULL,
  ignore_pretraining_limits = FALSE,
  ...
)

# S3 method for class 'recipe'
brulee_tab_pfn(
  x,
  data,
  num_estimators = NULL,
  softmax_temperature = NULL,
  balance_probabilities = FALSE,
  average_before_softmax = FALSE,
  training_set_limit = Inf,
  version = NULL,
  device = NULL,
  ignore_pretraining_limits = FALSE,
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

- ...:

  Not currently used, but required for extensibility.

- y:

  When `x` is a **data frame** or **matrix**, `y` is the outcome
  specified as:

  - A **data frame** with 1 numeric column.

  - A **matrix** with 1 numeric column.

  - A numeric **vector** for regression or a **factor** for
    classification.

- num_estimators:

  An integer for the ensemble size. When `NULL` (the default), the model
  version's recommended size is used (8 for v3.5 and v3, 4 for
  v3.5-fast; v3 uses more for data with many predictors).

- softmax_temperature:

  An adjustment factor that is a divisor in the exponents of the softmax
  function; it must be greater than 0. When `NULL` (the default), the
  model version's recommended value is used (1 for v3.5, 0.9 for v3).

- balance_probabilities:

  A logical to adjust the prior probabilities in cases where there is a
  class imbalance. Default is `FALSE`. Classification only.

- average_before_softmax:

  A logical. For cases where `num_estimators > 1`, should the average be
  done before using the softmax function or after? Default is `FALSE`.

- training_set_limit:

  An integer of at least 2, or `Inf` (the default) to use every row.
  When the training set is larger, it is sampled down to exactly that
  many rows, stratified by class for classification and by quartile for
  regression. For classification, it must be at least the number of
  classes, so that every class keeps a row.

- version:

  The model version, such as `"v3.5"`. A bare number works too: `3.5`,
  `"3.5"`, and `"v3.5"` are equivalent. See
  [`tab_pfn_versions()`](https://brulee.tidymodels.org/dev/reference/tab_pfn_versions.md)
  for the currently supported versions. The default is the newest
  version in that list.

- device:

  The torch device: `NULL` (the default; CUDA when available, otherwise
  the CPU), `"cpu"`, `"cuda"`, or `"mps"` (Apple GPUs).

- ignore_pretraining_limits:

  A logical. By default, the model refuses data larger than it was
  trained for (for example, more than 5,000 training rows on the CPU for
  v3.5). Set to `TRUE` to run it anyway.

- formula:

  A formula specifying the outcome terms on the left-hand side, and the
  predictor terms on the right-hand side.

- data:

  When a **recipe** or **formula** is used, `data` is specified as:

  - A **data frame** containing both the predictors and the outcome.

## Value

A `brulee_tab_pfn` object.

## Details

### Before you start: the model weights and their license

TabPFN is a *pre-trained* model. Instead of estimating parameters from
your data, `brulee_tab_pfn()` feeds your training data, together with
the rows to predict, to a large neural network that was trained in
advance on millions of synthetic data sets. That network's parameters
(its "weights") are made by the company Prior Labs and are **not**
included in brulee: you download them once, and Prior Labs requires that
you accept their license first.

The license allows free use for non-commercial purposes, such as
research, teaching, and evaluation. Commercial or production use
requires a separate license from Prior Labs (<sales@priorlabs.ai>). Read
the license when you accept it.

Each model version has its own license. Accepting the TabPFN-3.5 license
covers `"v3.5"` (the default) and `"v3.5-fast"`; `"v3"` needs the
TabPFN-3 license.

#### One-time setup

1.  Create a free account at <https://ux.priorlabs.ai>.

2.  On the **Licenses** tab, accept the license for each model version
    you plan to use (TabPFN-3.5 and/or TabPFN-3).

3.  On your account page, copy your **API key**. Treat it like a
    password.

4.  Give R the key by adding a line to your `.Renviron` file (open it
    with `usethis::edit_r_environ()`), then restart R:

        TABPFN_TOKEN=paste-your-api-key-here

5.  Download the weights for the version you will use:

        brulee::tab_pfn_download_weights("v3.5")

    This checks your key and license with Prior Labs, then downloads the
    weights from Hugging Face: about 880 MB for `"v3.5"`, 330 MB for
    `"v3.5-fast"`, and 450 MB for `"v3"` (one file each for
    classification and regression).
    [`tab_pfn_weights_available()`](https://brulee.tidymodels.org/dev/reference/tab_pfn_download_weights.md)
    tells you whether they are already downloaded.

In an interactive session you can skip steps 4 and 5: the first time
`brulee_tab_pfn()` needs weights that aren't downloaded, it offers to
download them, opens the Prior Labs login page in your browser, and asks
you to paste your API key. The key is then saved in
`~/.cache/tabpfn/auth_token` for later sessions. Non-interactive
sessions (scripts, R Markdown and Quarto documents, continuous
integration) need `TABPFN_TOKEN`; if the weights are missing there,
`brulee_tab_pfn()` stops with an error that says how to download them.

#### After the setup

The weights are stored in a per-user cache (see
[`tab_pfn_download_weights()`](https://brulee.tidymodels.org/dev/reference/tab_pfn_download_weights.md))
and used from there, with no further contact with Prior Labs or internet
access. Remove them with
[`tab_pfn_clear_cache()`](https://brulee.tidymodels.org/dev/reference/tab_pfn_clear_cache.md),
for example to free disk space, or because you stop using the model, as
the license requires when it ends.

If you have used the Python `tabpfn` package on this computer, it may
already have downloaded the weights and saved your API key; brulee uses
both, so you may have nothing to do.

### Computing requirements

The model runs on the CPU or, when available, on a CUDA GPU (or an Apple
GPU with `device = "mps"`). On the CPU it accepts at most 5,000 training
rows; see "Data size limits" below.

### Differences from the Python package

The computations follow the Python `tabpfn` package (version 9.1.0), but
the random choices of the ensemble members use R's random number
generator (so [`set.seed()`](https://rdrr.io/r/base/Random.html) makes
fits reproducible), and predictions differ slightly from Python's. For
v3, Python computes the SVD features with a randomized SVD when the
training data has more than a million cells; brulee always uses the
exact SVD.

Do not use `brulee_tab_pfn()` and the Python-based tabpfn package in the
same R session: R torch and Python torch can't be loaded in one process.

## Data size limits

Each model version was trained for data up to a certain size, and
`brulee_tab_pfn()` refuses larger data with an error that names the
count and the limit. Rows are training rows.

|               |            |            |            |         |
|---------------|------------|------------|------------|---------|
| Version       | Rows (GPU) | Rows (CPU) | Predictors | Classes |
| `"v3"`        | 1M         | 5K         | 2K         | 160     |
| `"v3.5"`      | 1M         | 5K         | 20K        | 160     |
| `"v3.5-fast"` | 1M         | 5K         | 20K        | 160     |

The CPU column applies whenever the model runs on the CPU (including
`device = NULL` on a machine without a CUDA GPU). Above a fifth of that
limit, `brulee_tab_pfn()` warns that prediction may be slow. Set
`ignore_pretraining_limits = TRUE` to lift the row and predictor limits;
the model then runs on data larger than it was trained for, more slowly
and possibly less accurately. The class limit can't be lifted. Use
`training_set_limit` to fit on a sample instead.

Memory grows with the number of rows times the number of predictors and
the ensemble size: on the CPU, 5,000 training rows with 50 predictors
and 8 ensemble members need about 12 GB. The row and predictor maxima
trade off against each other, so you cannot always reach both at once.
For `"v3.5"`, Prior Labs recommends up to 6,000 predictors even though
the model accepts 20,000. See <https://docs.priorlabs.ai/models>.

## References

Müller, S., Hollmann, N., Pineda Arango, S., Grabocka, J., and Hutter,
F. (2022). "Transformers can do Bayesian inference." *International
Conference on Learning Representations 2022*.
[doi:10.48550/arXiv.2112.10510](https://doi.org/10.48550/arXiv.2112.10510)

Hollmann, N., Müller, S., Eggensperger, K., and Hutter, F. (2023).
"TabPFN: A transformer that solves small tabular classification problems
in a second." *International Conference on Learning Representations
2023*.
[doi:10.48550/arXiv.2207.01848](https://doi.org/10.48550/arXiv.2207.01848)

Hollmann, N., Müller, S., Purucker, L., Krishnakumar, A., Körfer, M.,
Hoo, S. B., Schirrmeister, R. T., and Hutter, F. (2025). "Accurate
predictions on small data with a tabular foundation model." *Nature*,
637(8045), 319-326.
[doi:10.1038/s41586-024-08328-6](https://doi.org/10.1038/s41586-024-08328-6)

Grinsztajn, L., Flöge, K., Key, O., et al. (2026). "TabPFN-3: Technical
report." *arXiv preprint*.
[doi:10.48550/arXiv.2605.13986](https://doi.org/10.48550/arXiv.2605.13986)

Jäger, B., Erickson, N., Grinsztajn, L., et al. (2026). "TabPFN-3.5:
Technical report." *arXiv preprint*.
[doi:10.48550/arXiv.2609.17895](https://doi.org/10.48550/arXiv.2609.17895)

## See also

[`predict.brulee_tab_pfn()`](https://brulee.tidymodels.org/dev/reference/predict.brulee_tab_pfn.md),
[`tab_pfn_download_weights()`](https://brulee.tidymodels.org/dev/reference/tab_pfn_download_weights.md)

## Examples

``` r
if (FALSE) { # \dontrun{
if (rlang::is_installed("modeldata") && tab_pfn_weights_available()) {

  # --------------------------------------------------------------------------
  # Regression

  set.seed(1)
  reg_fit <- brulee_tab_pfn(mpg ~ ., data = mtcars[6:32,])
  reg_fit

  augment(reg_fit, new_data =  mtcars[1:5, -1])
  augment(reg_fit, new_data =  mtcars[1:5, -1], quantile_levels = (1:5)/6)

  # --------------------------------------------------------------------------
  # Classification

  set.seed(1)
  cls_fit <- brulee_tab_pfn(Species ~ ., data = modeldata::scat[6:110,])
  cls_fit

  augment(cls_fit, new_data =  modeldata::scat[1:5, -1])
}
} # }
```
