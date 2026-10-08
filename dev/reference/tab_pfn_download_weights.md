# Download and cache pretrained TabPFN weights

[`brulee_tab_pfn()`](https://brulee.tidymodels.org/dev/reference/brulee_tab_pfn.md)
needs pretrained weights that are not shipped with the package.
`tab_pfn_download_weights()` downloads them from Prior Labs' Hugging
Face repositories into the local cache. `tab_pfn_weights_available()`
reports whether the cache already holds them.

## Usage

``` r
tab_pfn_download_weights(
  version = NULL,
  task = c("classification", "regression"),
  files = c("default", "all"),
  cache_dir = tabpfn_cache_dir()
)

tab_pfn_weights_available(
  version = NULL,
  task = c("classification", "regression"),
  cache_dir = tabpfn_cache_dir()
)
```

## Arguments

- version:

  A model version from
  [`tab_pfn_versions()`](https://brulee.tidymodels.org/dev/reference/tab_pfn_versions.md).
  A number such as `3.5` is also accepted. The default is the newest
  version.

- task:

  The task(s), one or both of `"classification"` and `"regression"`.
  Both by default. (v3.5 uses one checkpoint for both.)

- files:

  `"default"` downloads each task's default checkpoint; `"all"` also
  downloads the alternative checkpoints.

- cache_dir:

  The root of the local weight cache.

## Value

`tab_pfn_download_weights()` invisibly returns the paths of the
checkpoints. `tab_pfn_weights_available()` returns a single logical:
`TRUE` when the checkpoints for every task in `task` are cached.

## Details

Files already cached (by brulee, or by the Python `tabpfn` package) are
checked and skipped, so re-running does not download again. Attaching
brulee never downloads the weights. If
[`brulee_tab_pfn()`](https://brulee.tidymodels.org/dev/reference/brulee_tab_pfn.md)
is run before they are cached, it asks to download them in an
interactive session and errors, pointing here, otherwise.

`tab_pfn_weights_available()` only looks for files of the expected size;
[`brulee_tab_pfn()`](https://brulee.tidymodels.org/dev/reference/brulee_tab_pfn.md)
verifies their checksums when it first loads them.

## Cache location

The weights are stored in a `tabpfn` directory of brulee's per-user
cache,
[`tools::R_user_dir()`](https://rdrr.io/r/tools/userdir.html)`("brulee", "cache")`.
Set the `brulee.tabpfn_cache_dir` option to use another directory.
Weights that the Python `tabpfn` package has already downloaded (into
its own cache, or the directory in the `TABPFN_MODEL_CACHE_DIR`
environment variable) are used from there without copying, after their
checksum is verified. Use
[`tab_pfn_clear_cache()`](https://brulee.tidymodels.org/dev/reference/tab_pfn_clear_cache.md)
to delete the weights brulee downloaded.

Notes about where the weights come from (downloaded, already cached, or
reused from the Python package) appear only in interactive sessions, at
most once a day for each file.

## License

The weights are released by Prior Labs under a non-commercial license
that must be accepted once before the first download. The first download
asks you to log in to Prior Labs, accept the license, and paste your API
key. The key is stored in `~/.cache/tabpfn/auth_token`, the same file
the Python `tabpfn` package uses. In non-interactive sessions, set the
`TABPFN_TOKEN` environment variable to your API key instead. See
[`brulee_tab_pfn()`](https://brulee.tidymodels.org/dev/reference/brulee_tab_pfn.md)
for the full setup.

## Examples

``` r
if (FALSE) {
tab_pfn_download_weights("v3.5")
tab_pfn_weights_available("v3.5")
tab_pfn_weights_available("v3", task = "regression")
}
```
