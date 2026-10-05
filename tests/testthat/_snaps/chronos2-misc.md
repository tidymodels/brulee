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

# chronos2_resolve_revision errors when sha is empty string

    Code
      brulee:::chronos2_resolve_revision("amazon/chronos-2", "main")
    Condition
      Error in `brulee:::chronos2_resolve_revision()`:
      ! HuggingFace API did not return a SHA for revision "main".

