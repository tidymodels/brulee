# chronos2_resolve_revision errors on a non-200 status code

    Code
      brulee:::chronos2_resolve_revision("amazon/chronos-2", "v0")
    Condition
      Error in `brulee:::chronos2_resolve_revision()`:
      ! Failed to resolve revision "v0" for "amazon/chronos-2" (HTTP 404).

# chronos2_resolve_revision errors when curl itself fails

    Code
      brulee:::chronos2_resolve_revision("amazon/chronos-2", "v0")
    Condition
      Error in `brulee:::chronos2_resolve_revision()`:
      ! Failed to resolve revision "v0" for "amazon/chronos-2".
      x simulated network failure

# chronos2_resolve_revision errors when HF API has no sha field

    Code
      brulee:::chronos2_resolve_revision("amazon/chronos-2", "v0")
    Condition
      Error in `brulee:::chronos2_resolve_revision()`:
      ! HuggingFace API did not return a SHA for revision "v0".

# chronos2_download_file errors after exhausting retries

    Code
      brulee:::chronos2_download_file("http://x", tmp, "test", max_attempts = 2L)
    Message
      i Downloading <http://x>
      ! Attempt 1/2 for "test" failed; retrying.
      i Downloading <http://x>
      v Downloading <http://x> [TIME]
      
      i Downloading <http://x>
    Condition
      Error in `brulee:::chronos2_download_file()`:
      ! Failed to download <http://x> after 2 attempts.
      i If you keep hitting this, try a different network or proxy.
    Message
      x Downloading <http://x> [TIME]
      

