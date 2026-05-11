if (requireNamespace("spelling", quietly = TRUE) &&
    !identical(Sys.getenv("R_COVR"), "true"))
  spelling::spell_check_test(vignettes = TRUE, error = FALSE,
                             skip_on_cran = TRUE)
