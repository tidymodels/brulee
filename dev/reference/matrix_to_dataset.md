# Convert data to torch format

For an x/y interface, `matrix_to_dataset()` converts the data to proper
encodings then formats the results for consumption by `torch`.

## Usage

``` r
matrix_to_dataset(x, y, device = NULL)
```

## Arguments

- x:

  A numeric matrix of predictors.

- y:

  A vector. If regression than `y` is numeric. For classification, it is
  a factor.

- device:

  A single character string for the device to use (e.g., `"cpu"` or
  `"cuda"`). The default of `NULL` uses the CPU. See
  [training_efficiency](https://brulee.tidymodels.org/dev/reference/training_efficiency.md).

## Value

An R6 index sampler object with classes "training_set", "dataset", and
"R6".

## Details

Missing values should be removed before passing data to this function.

## Examples

``` r
if (torch::torch_is_installed()) {
  matrix_to_dataset(as.matrix(mtcars[, -1]), mtcars$mpg)
}
#> <tensor_dataset>
#>   Inherits from: <dataset>
#>   Public:
#>     .getbatch: function (index) 
#>     .getitem: function (index, ..., drop = TRUE) 
#>     .length: function () 
#>     clone: function (deep = FALSE) 
#>     initialize: function (...) 
#>     load_state_dict: function (x, ..., .refer_to_state_dict = FALSE) 
#>     state_dict: function () 
#>     tensors: list
```
