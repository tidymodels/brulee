# Add model predictions to data

[`augment()`](https://generics.r-lib.org/reference/augment.html) adds
every prediction that a model can make for its mode to `new_data`. It is
a convenience wrapper around
[`predict()`](https://rdrr.io/r/stats/predict.html) and is a common
first step before computing performance metrics with the yardstick
package.

## Usage

``` r
# S3 method for class 'brulee_mlp'
augment(x, new_data, ...)

# S3 method for class 'brulee_linear_reg'
augment(x, new_data, ...)

# S3 method for class 'brulee_logistic_reg'
augment(x, new_data, ...)

# S3 method for class 'brulee_multinomial_reg'
augment(x, new_data, ...)

# S3 method for class 'brulee_resnet'
augment(x, new_data, ...)

# S3 method for class 'brulee_rln'
augment(x, new_data, ...)

# S3 method for class 'brulee_saint'
augment(x, new_data, ...)

# S3 method for class 'brulee_auto_int'
augment(x, new_data, ...)

# S3 method for class 'brulee_tab_icl'
augment(x, new_data, quantile_levels = NULL, ...)

# S3 method for class 'brulee_chronos'
augment(x, new_data = NULL, ...)
```

## Arguments

- x:

  A model fit from brulee.

- new_data:

  A data frame or matrix of predictors. The outcome column may also be
  present.

- ...:

  Options to pass to
  [`predict()`](https://rdrr.io/r/stats/predict.html), such as `epoch`.

- quantile_levels:

  A numeric vector of quantile levels, each in the open interval
  `(0, 1)`, sorted and unique. Only used by
  [`brulee_tab_icl()`](https://brulee.tidymodels.org/dev/reference/brulee_tab_icl.md)
  regression fits. The default of `NULL` gives an ordinary numeric
  prediction; supplying levels additionally returns the predictive
  distribution. See the details.

## Value

A tibble with the same number of rows as `new_data`, with the prediction
columns described above prepended to the columns of `new_data`.

## Details

For regression models, a `.pred` column is added. When the outcome
column is also in `new_data`, a `.resid` column of residuals (i.e.,
outcome minus prediction) is added as well.

For classification models, a `.pred_class` column of hard class
predictions is added along with one `.pred_{level}` column of class
probabilities per outcome level. No residuals are computed.

For
[`brulee_chronos()`](https://brulee.tidymodels.org/dev/reference/brulee_chronos.md)
models, `new_data` describes the future window to forecast for and is
required. The columns added are those that
[`predict()`](https://rdrr.io/r/stats/predict.html) would return for the
given `type`: `.pred`, `.pred_quantile`, or both. A `.resid` column is
added when the outcome column is in `new_data`, as it is when the
forecast is being compared against a known future.

In all cases the new columns are added to the front of `new_data`.
`new_data` is validated by
[`predict()`](https://rdrr.io/r/stats/predict.html), so the same columns
are required as for prediction.

### Distributional predictions for [`brulee_tab_icl()`](https://brulee.tidymodels.org/dev/reference/brulee_tab_icl.md) regression fits

The regression head of a
[`brulee_tab_icl()`](https://brulee.tidymodels.org/dev/reference/brulee_tab_icl.md)
model is a quantile regression head, so it always has a full predictive
distribution available rather than just a point estimate. By default
[`augment()`](https://generics.r-lib.org/reference/augment.html) ignores
that and returns an ordinary numeric prediction, exactly like every
other regression model: `.pred` and `.resid`.

Setting `quantile_levels` asks for the distribution as well. The result
then also has a `.pred_quantile` column (a
[`hardhat::quantile_pred()`](https://hardhat.tidymodels.org/reference/quantile_pred.html)
vector at the requested levels) and a `.pred_variance` column. These are
further readouts of the same distribution as the mean, so they are added
*alongside* `.pred` rather than in place of it.

Requesting the distribution also changes what `.resid` is measured
against. `.pred` is still the mean, but the residual is taken against
the distribution's **median**: the head can return a skewed
distribution, and for a skewed one the mean is pulled toward the long
tail while the median stays with the bulk of the mass, which makes the
median the more representative point estimate to measure residuals
against. So `.resid` is `outcome - .pred` when `quantile_levels` is
`NULL` and `outcome - median` when it is set.
([`brulee_chronos()`](https://brulee.tidymodels.org/dev/reference/brulee_chronos.md)
defines `.pred` to be the median already, so its residuals are against
the median in either case.)

The median is read off `.pred_quantile` when `quantile_levels` includes
`0.5`, and is otherwise requested on its own. That extra readout only
happens when `new_data` actually carries the outcome column, so scoring
data without an outcome costs nothing more.

## See also

[`predict.brulee_mlp()`](https://brulee.tidymodels.org/dev/reference/predict.brulee_mlp.md)

## Examples

``` r
# \donttest{
if (torch::torch_is_installed() && rlang::is_installed(c("recipes", "modeldata"))) {

 library(recipes)

 # ---------------------------------------------------------------------------
 # regression

 data(ames, package = "modeldata")

 ames$Sale_Price <- log10(ames$Sale_Price)

 set.seed(1)
 in_train <- sample(seq_len(nrow(ames)), 2000)
 ames_train <- ames[ in_train,]
 ames_test  <- ames[-in_train,]

 ames_rec <-
  recipe(Sale_Price ~ Longitude + Latitude, data = ames_train) |>
    step_normalize(all_numeric_predictors())

 set.seed(2)
 reg_fit <- brulee_mlp(ames_rec, data = ames_train, epochs = 50, batch_size = 32)

 # `.pred` and `.resid` are added to the front of the data:
 augment(reg_fit, ames_test)

 # Without the outcome column, there are no residuals:
 augment(reg_fit, ames_test[, c("Longitude", "Latitude")])

 # Predictions from a specific epoch:
 augment(reg_fit, ames_test, epoch = 1)

 # ---------------------------------------------------------------------------
 # classification

 data(penguins, package = "modeldata")

 penguins <- penguins |> na.omit()

 set.seed(3)
 in_train <- sample(seq_len(nrow(penguins)), 200)
 penguins_train <- penguins[ in_train,]
 penguins_test  <- penguins[-in_train,]

 penguins_rec <-
  recipe(species ~ ., data = penguins_train) |>
    step_dummy(all_nominal_predictors()) |>
    step_normalize(all_numeric_predictors())

 set.seed(4)
 cls_fit <- brulee_mlp(penguins_rec, data = penguins_train, epochs = 20)

 # `.pred_class` plus one probability column per class:
 augment(cls_fit, penguins_test)
}
#> Loading required package: dplyr
#> 
#> Attaching package: ‘dplyr’
#> The following objects are masked from ‘package:stats’:
#> 
#>     filter, lag
#> The following objects are masked from ‘package:base’:
#> 
#>     intersect, setdiff, setequal, union
#> 
#> Attaching package: ‘recipes’
#> The following object is masked from ‘package:stats’:
#> 
#>     step
#> # A tibble: 133 × 11
#>    .pred_class .pred_Adelie .pred_Chinstrap .pred_Gentoo species island
#>    <fct>              <dbl>           <dbl>        <dbl> <fct>   <fct> 
#>  1 Adelie             0.663           0.337     7.88e-11 Adelie  Torge…
#>  2 Adelie             0.663           0.337     7.88e-11 Adelie  Torge…
#>  3 Adelie             0.663           0.337     7.88e-11 Adelie  Torge…
#>  4 Adelie             0.663           0.337     7.88e-11 Adelie  Biscoe
#>  5 Adelie             0.663           0.337     7.88e-11 Adelie  Biscoe
#>  6 Adelie             0.663           0.337     7.88e-11 Adelie  Biscoe
#>  7 Adelie             0.663           0.337     7.88e-11 Adelie  Biscoe
#>  8 Adelie             0.663           0.337     7.88e-11 Adelie  Biscoe
#>  9 Adelie             0.663           0.337     7.88e-11 Adelie  Biscoe
#> 10 Adelie             0.663           0.337     7.88e-11 Adelie  Biscoe
#> # ℹ 123 more rows
#> # ℹ 5 more variables: bill_length_mm <dbl>, bill_depth_mm <dbl>,
#> #   flipper_length_mm <int>, body_mass_g <int>, sex <fct>
# }
```
