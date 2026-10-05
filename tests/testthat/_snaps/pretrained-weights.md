# brulee_confirm_download errors when non-interactive

    Code
      brulee_confirm_download(label = "amazon/chronos-2", size = "500MB", fn = "brulee_chronos",
        root = root, hint = "Run {.fn brulee_chronos} in an interactive session to download them.")
    Condition
      Error:
      ! No cached amazon/chronos-2 weights found in '<root>'.
      i Run `brulee_chronos()` in an interactive session to download them.

# brulee_confirm_download aborts when the user declines

    Code
      brulee_confirm_download(label = "amazon/chronos-2", size = "500MB", fn = "brulee_chronos",
        root = tempdir(), hint = "Run {.fn brulee_chronos} in an interactive session to download them.")
    Message
      The weights for amazon/chronos-2 are not found locally.
    Condition
      Error:
      ! Download declined; `brulee_chronos()` needs the weights to continue.

# brulee_download_file errors after exhausting retries

    Code
      brulee:::brulee_download_file("http://x", tmp, "test", max_attempts = 2L)
    Message
      i Downloading <http://x>
      ! Attempt 1/2 for "test" failed; retrying.
      i Downloading <http://x>
      v Downloading <http://x> [TIME]
      
      i Downloading <http://x>
    Condition
      Error in `brulee:::brulee_download_file()`:
      ! Failed to download <http://x> after 2 attempts.
      i If you keep hitting this, try a different network or proxy.
    Message
      x Downloading <http://x> [TIME]
      

# brulee_download_file rejects a download with the wrong checksum

    Code
      suppressMessages(brulee_download_file(remote$url, dest, label = "model.safetensors",
      size = remote$size, sha256 = strrep("0", 64)))
    Condition
      Error in `brulee_download_file()`:
      ! The downloaded file 'model.safetensors' does not have the expected checksum.
      i The upstream file may have changed; please report this at <https://github.com/tidymodels/brulee/issues>.

