# model versions are normalized and checked

    Code
      tabpfn_resolve_version(2.5)
    Condition
      Error:
      ! `2.5` must be one of "v3", "v3.5", or "v3.5-fast".
      x "v2.5" is not a supported model version.

---

    Code
      tabpfn_resolve_version(c("v3", "v3.5"))
    Condition
      Error:
      ! `c("v3", "v3.5")` must be a single string or number, not a character vector.

# version info names the checkpoint and its architecture

    Code
      tabpfn_version_info("v3", "regression", "tabpfn-v3-classifier-v3_default.ckpt")
    Condition
      Error:
      ! "tabpfn-v3-classifier-v3_default.ckpt" is not a regression checkpoint for TabPFN v3.
      i Available files: "tabpfn-v3-regressor-v3_default.ckpt", "tabpfn-v3-regressor-v3_20260417_mediumdata.ckpt", "tabpfn-v3-regressor-v3_20260506_timeseries.ckpt", and "tabpfn-v3-regressor-v3_20260506_ood.ckpt".

# cached checkpoints are checked, and missing ones need consent

    Code
      tabpfn_checkpoint_path(info)
    Condition
      Error:
      ! No cached TabPFN v3.5 weights found in '<cache>'.
      i Download them with `brulee::tab_pfn_download_weights("v3.5")`.

---

    Code
      tabpfn_checkpoint_path(bad_size)
    Condition
      Error:
      ! The cached checkpoint '<cache>/model.safetensors' has an unexpected size (200 B, expected 10 B).
      i It may be incomplete or a different release. Delete it and download it again.

---

    Code
      tabpfn_checkpoint_path(bad_sha)
    Condition
      Error:
      ! The cached checkpoint '<cache>/model.safetensors' does not have the expected checksum.
      i It may be corrupted or a different release. Delete it and download it again.

# downloading checks the license first

    Code
      path <- tabpfn_checkpoint_path(info, ask = FALSE)
    Message
      i Downloading the TabPFN v3.5 weights ('model.safetensors', 200 B) from <https://huggingface.co/Prior-Labs/tabpfn_3_5> into '<cache>'.
      i Downloading <remote>
      v Downloading <remote>
      
      v Saved the TabPFN v3.5 weights to '<cache>/model.safetensors'.

# weights cached by Python are used in place

    Code
      path <- tabpfn_checkpoint_path(info, cache_dir = cache)
    Message
      i Using the TabPFN v3.5 weights downloaded by the Python tabpfn package: '<python-cache>/model.safetensors'.

# checking for weights needs a known version and task

    Code
      tab_pfn_weights_available("v2")
    Condition
      Error in `tab_pfn_weights_available()`:
      ! `version` must be one of "v3", "v3.5", or "v3.5-fast".
      x "v2" is not a supported model version.

---

    Code
      tab_pfn_weights_available("v3.5", task = "survival")
    Condition
      Error in `tab_pfn_weights_available()`:
      ! `task` must be one of "classification" or "regression", not "survival".

# the cache can be cleared, by version

    Code
      deleted <- tab_pfn_clear_cache("v3", cache_dir = cache, ask = FALSE)
    Message
      v Deleted 2 cached TabPFN v3 weight files (20 B) from '<cache>'.

---

    Code
      tab_pfn_clear_cache(cache_dir = cache, ask = FALSE)
    Message
      v Deleted 1 cached TabPFN weight file (10 B) from '<cache>'.

---

    Code
      tab_pfn_clear_cache(cache_dir = cache, ask = FALSE)
    Message
      i No cached TabPFN weights in '<cache>'.

# clearing the cache asks first in interactive sessions

    Code
      tab_pfn_clear_cache(cache_dir = cache, ask = TRUE)
    Message
      These cached TabPFN weight file (10 B) will be deleted:
      * tabpfn-v3.5-20260909.safetensors
      Nothing was deleted.

# license problems give informative errors

    Code
      tabpfn_ensure_license("tabpfn_3_5")
    Condition
      Error:
      ! TabPFN needs a one-time license acceptance before the model weights can be downloaded.
      i In a non-interactive session:
      * Log in (or register) at <https://ux.priorlabs.ai>.
      * Accept the license on the Licenses tab.
      * Copy your API key from <https://ux.priorlabs.ai/account>.
      * Set the `TABPFN_TOKEN` environment variable to it, e.g. in your '.Renviron' file.

---

    Code
      tabpfn_ensure_license("tabpfn_3_5")
    Condition
      Error:
      ! Could not reach the Prior Labs license server.
      i Check your internet connection and try again.

---

    Code
      tabpfn_ensure_license("tabpfn_3_5")
    Message
      ! The API key in `TABPFN_TOKEN` is invalid or expired.
    Condition
      Error:
      ! TabPFN needs a one-time license acceptance before the model weights can be downloaded.
      i In a non-interactive session:
      * Log in (or register) at <https://ux.priorlabs.ai>.
      * Accept the license on the Licenses tab.
      * Copy your API key from <https://ux.priorlabs.ai/account>.
      * Set the `TABPFN_TOKEN` environment variable to it, e.g. in your '.Renviron' file.

# a pasted API key is verified and saved

    Code
      tabpfn_ensure_license("tabpfn_3_5")
    Message
      v API key saved to '<tmp>/cache/auth_token'.

