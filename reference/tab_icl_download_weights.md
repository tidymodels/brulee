# Download and cache pretrained TabICL weights

[`brulee_tab_icl()`](https://brulee.tidymodels.org/reference/brulee_tab_icl.md)
needs pretrained weights that are not shipped with the package.
`tab_icl_download_weights()` fetches them from a release of the
`tidymodels/tabicl-weights` GitHub repository into the local cache.
`tab_icl_weights_available()` reports whether the cache already holds
usable weights.

## Usage

``` r
tab_icl_download_weights(
  task = c("classification", "regression"),
  version = tabicl_default_version(),
  date = tabicl_default_date(),
  repo = tabicl_default_repo(),
  cache_dir = tabicl_cache_dir(),
  call = rlang::caller_env()
)

tab_icl_weights_available(
  task = c("classification", "regression"),
  cache_dir = tabicl_cache_dir()
)
```

## Arguments

- task:

  The task(s) to act on, one or both of `"classification"` and
  `"regression"`. Both are fetched (and checked) by default.

- version, date:

  The release to fetch, identifying the tag `<version>-<date>` (for
  example `"v2"` and `"2026-02-12"`).

- repo:

  The `owner/name` of the GitHub repository hosting the weights.

- cache_dir:

  The root of the local weight cache. Defaults to the
  `brulee.tabicl_cache_dir` option, or a per-user cache directory via
  [`tools::R_user_dir()`](https://rdrr.io/r/tools/userdir.html)`("brulee", "cache")`.

- call:

  The calling environment, used for error messages.

## Value

`tab_icl_download_weights()` invisibly returns the populated
`<cache_dir>/<version>/<date>` directory. `tab_icl_weights_available()`
returns a single logical.

## Details

Each release carries the two files brulee reads per task (a JSON config
and a safetensors weight file) as individual assets. They are downloaded
into `<cache_dir>/<version>/<date>/<TaskLabel>/`. A file already present
and complete is left in place, so re-running resumes rather than
re-downloads.

The cache location can be overridden with the `brulee.tabicl_cache_dir`
option. Attaching brulee never downloads the weights. If
[`brulee_tab_icl()`](https://brulee.tidymodels.org/reference/brulee_tab_icl.md)
is run before they are cached, it prompts to download them (via this
function) in an interactive session and errors, pointing you here,
otherwise.

## Examples

``` r
if (FALSE) {
tab_icl_download_weights()
tab_icl_weights_available()
}
```
