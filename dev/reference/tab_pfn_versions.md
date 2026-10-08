# List available TabPFN model versions

Returns the model versions that
[`brulee_tab_pfn()`](https://brulee.tidymodels.org/dev/reference/brulee_tab_pfn.md)
can run, which can be passed to its `version` argument and to
[`tab_pfn_download_weights()`](https://brulee.tidymodels.org/dev/reference/tab_pfn_download_weights.md).

## Usage

``` r
tab_pfn_versions()
```

## Value

A character vector of model version strings.

## Examples

``` r
tab_pfn_versions()
#> [1] "v3"        "v3.5"      "v3.5-fast"
```
