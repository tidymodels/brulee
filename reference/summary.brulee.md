# Summarize the architecture of a brulee model

[`summary()`](https://rdrr.io/r/base/summary.html) methods brulee neural
network models print a layer-by-layer description of the fitted torch
module: each component's type, shape, and parameter count, followed by
the total parameter count. For `brulee_resnet`, residual (skip)
connections and their projection layers are shown at the block
boundaries where they apply.

## Usage

``` r
# S3 method for class 'brulee_mlp'
summary(object, ...)

# S3 method for class 'brulee_resnet'
summary(object, ...)

# S3 method for class 'brulee_rln'
summary(object, ...)

# S3 method for class 'brulee_auto_int'
summary(object, ...)

# S3 method for class 'brulee_saint'
summary(object, ...)
```

## Arguments

- object:

  A `brulee_resnet`, `brulee_mlp`, `brulee_rln`, `brulee_auto_int`, or
  `brulee_saint` object.

- ...:

  Not used.

## Value

The model object, invisibly. Called for its side effect of printing the
architecture.

## Examples

``` r
# \donttest{
if (torch::torch_is_installed() & rlang::is_installed("modeldata")) {
  data(ames, package = "modeldata")
  ames$Sale_Price <- log10(ames$Sale_Price)

  set.seed(1)
  fit <- brulee_resnet(Sale_Price ~ Longitude + Latitude, data = ames,
                       hidden_units = c(8, 4), bottleneck_units = c(6, 3),
                       residual_at = 2, epochs = 3)
  summary(fit)
}
#> Residual network architecture
#> inputs: 2 | output dim: 1 | layers: 2
#> 
#> Residual group 1 (blocks 1-2, + skip)
#>   Block 1:
#>     BatchNorm1d(2)                        4 params
#>     Linear(2 -> 6)                       18 params
#>     ReLU                                  0 params
#>     Linear(6 -> 8)                       56 params
#>   Block 2:
#>     BatchNorm1d(8)                       16 params
#>     Linear(8 -> 3)                       27 params
#>     ReLU                                  0 params
#>     Linear(3 -> 4)                       16 params
#>   + skip: Linear(2 -> 4)                 12 params
#> 
#> Output head
#>     BatchNorm1d(4)                        8 params
#>     Linear(4 -> 1)                        5 params
#> 
#> Total parameters: 162
# }
```
