# Extract Model Coefficients

Extract Model Coefficients

## Usage

``` r
# S3 method for class 'brulee_logistic_reg'
coef(object, epoch = NULL, ...)

# S3 method for class 'brulee_linear_reg'
coef(object, epoch = NULL, ...)

# S3 method for class 'brulee_mlp'
coef(object, epoch = NULL, ...)

# S3 method for class 'brulee_multinomial_reg'
coef(object, epoch = NULL, ...)

# S3 method for class 'brulee_resnet'
coef(object, epoch = NULL, ...)

# S3 method for class 'brulee_rln'
coef(object, epoch = NULL, ...)
```

## Arguments

- object:

  A model fit from brulee.

- epoch:

  A single integer for the training iteration. If left `NULL`, the
  estimates from the best model fit (via internal performance metrics).

- ...:

  Not currently used.

## Value

For logistic/linear regression, a named vector. For neural networks, a
list of arrays.

## Examples

``` r
# \donttest{
if (torch::torch_is_installed() && rlang::is_installed(c("recipes", "modeldata"))) {

 data(ames, package = "modeldata")

 ames$Sale_Price <- log10(ames$Sale_Price)

 set.seed(1)
 in_train <- sample(seq_len(nrow(ames)), 2000)
 ames_train <- ames[ in_train,]
 ames_test  <- ames[-in_train,]

 # Using recipe
 library(recipes)

 ames_rec <-
  recipe(Sale_Price ~ Longitude + Latitude, data = ames_train) |>
    step_normalize(all_numeric_predictors())

 set.seed(2)
 fit <- brulee_linear_reg(ames_rec, data = ames_train, epochs = 50)

 coef(fit)
 coef(fit, epoch = 1)
}
#> (Intercept)   Longitude    Latitude 
#>  5.22475100 -0.05370786  0.05004638 
# }
```
