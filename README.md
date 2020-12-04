
<!-- README.md is generated from README.Rmd. Please edit that file -->

# lantern

<!-- badges: start -->

[![Codecov test
coverage](https://codecov.io/gh/topepo/lantern/branch/master/graph/badge.svg)](https://codecov.io/gh/topepo/lantern?branch=master)
[![Lifecycle: experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://www.tidyverse.org/lifecycle/#experimental)
<!-- badges: end -->

`lantern` packages several basic modeling functions that use the `torch`
package.

The package is currently experimental; the user interface and other
details may change before release.

## Installation

You can install the released version of lantern from
[CRAN](https://CRAN.R-project.org) with:

``` r
install.packages("lantern")
```

And the development version from [GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("topepo/lantern")
```

## Example

`lantern` has formula, x/y, and recipe user interfaces for each
function. For example:

``` r
library(lantern)
library(recipes)
library(yardstick)

data(bivariate, package = "modeldata")
set.seed(20)
nn_log_biv <- lantern_mlp(Class ~ log(A) + log(B), data = bivariate_train, 
                          epochs = 150, hidden_units = 3, batch_size = 64)

# We use the tidymodels semantics to always return a tibble when predicting
predict(nn_log_biv, bivariate_test, type = "prob") %>% 
  bind_cols(bivariate_test) %>% 
  roc_auc(Class, .pred_One)
#> # A tibble: 1 x 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 roc_auc binary         0.834
```

A recipe can also be used if the data require some sort of preprocessing
(e.g., indicator variables, transformations, or standardization):

``` r
library(recipes)

rec <- 
  recipe(Class ~ ., data = bivariate_train) %>%  
  step_YeoJohnson(all_predictors()) %>% 
  step_normalize(all_predictors())

set.seed(20)
nn_rec_biv <- lantern_mlp(rec, data = bivariate_train, 
                          epochs = 150, hidden_units = 3, batch_size = 64)

# A little better
predict(nn_rec_biv, bivariate_test, type = "prob") %>% 
  bind_cols(bivariate_test) %>% 
  roc_auc(Class, .pred_One)
#> # A tibble: 1 x 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 roc_auc binary         0.859
```

## Code of Conduct

Please note that the lantern project is released with a [Contributor
Code of
Conduct](https://contributor-covenant.org/version/2/0/CODE_OF_CONDUCT.html).
By contributing to this project, you agree to abide by its terms.
