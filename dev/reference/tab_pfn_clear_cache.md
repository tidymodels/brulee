# Remove cached TabPFN weights

Deletes checkpoints that brulee downloaded for
[`brulee_tab_pfn()`](https://brulee.tidymodels.org/dev/reference/brulee_tab_pfn.md),
for one model version or all of them, along with any partial downloads,
and unloads models loaded in the current session. Use it to free disk
space (each checkpoint takes 200 MB to 900 MB), to force a fresh
download, or to delete the weights when you stop using them, as the
Prior Labs license requires when it ends.

## Usage

``` r
tab_pfn_clear_cache(
  version = NULL,
  cache_dir = tabpfn_cache_dir(),
  ask = rlang::is_interactive()
)
```

## Arguments

- version:

  A model version from
  [`tab_pfn_versions()`](https://brulee.tidymodels.org/dev/reference/tab_pfn_versions.md),
  or `NULL` (the default) for all versions.

- cache_dir:

  The root of the local weight cache.

- ask:

  A logical: ask for confirmation before deleting? Defaults to `TRUE` in
  interactive sessions.

## Value

The paths of the deleted files, invisibly.

## Details

Weights downloaded by the Python `tabpfn` package are never removed:
delete those files yourself if you no longer need them.

## Examples

``` r
if (FALSE) {
tab_pfn_clear_cache("v3")
tab_pfn_clear_cache()
}
```
