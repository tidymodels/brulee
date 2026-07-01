# Tests for the TabICL weight loader (R/tabicl-load.R): the state_dict -> R
# parameter map, the both-direction "no skips" verification, config parsing, and
# the end-to-end load + forward round-trip. Uses the committed full_model
# fixtures (whose tensors are the full state dict keyed by the production
# prefixes, plus the inputs and golden output).

# Weight keys in the full_model fixture (everything except X / y_train / out).
tabicl_fixture_weight_keys <- function(f) {
  grep(
    "^(col_embedder|row_interactor|icl_predictor)\\.",
    names(f),
    value = TRUE
  )
}

for (fixture in c("full_model", "full_model_reg")) {
  local({
    fixture_name <- fixture
    test_that(
      paste0(
        "load_tabicl_weights maps the full state dict (",
        fixture_name,
        ")"
      ),
      {
        skip_if_no_tabicl_fixtures(fixture_name)

        f <- tabicl_load_fixture(fixture_name)
        meta <- tabicl_fixture_meta(fixture_name)

        model <- brulee:::tabicl_model(meta$config)
        tensors <- f[tabicl_fixture_weight_keys(f)]

        # Should map every tensor with no missing / unmatched parameter.
        expect_no_error(brulee:::load_tabicl_weights(model, tensors))

        model$eval()
        out <- model(f$X, f$y_train)
        expect_lt(tabicl_max_abs_diff(out, f$out), 1e-5)
      }
    )
  })
}

test_that("tabicl_load_model round-trips config + safetensors from disk", {
  skip_if_no_tabicl_fixtures("full_model")
  skip_if_not_installed("jsonlite")

  f <- tabicl_load_fixture("full_model")
  meta <- tabicl_fixture_meta("full_model")

  # Writes the task-prefixed files (full_model is a classification checkpoint).
  dir <- tabicl_write_model_dir(f, meta)

  loaded <- brulee:::tabicl_load_model(dir, task = "classification")
  out <- loaded$model(f$X, f$y_train)
  expect_lt(tabicl_max_abs_diff(out, f$out), 1e-5)
})

test_that("load_tabicl_weights errors on unmatched and missing keys", {
  skip_if_no_tabicl_fixtures("full_model")

  f <- tabicl_load_fixture("full_model")
  meta <- tabicl_fixture_meta("full_model")
  model <- brulee:::tabicl_model(meta$config)
  tensors <- f[tabicl_fixture_weight_keys(f)]

  # Extra (unmatched) key.
  extra <- tensors
  extra[["col_embedder.bogus.weight"]] <- tensors[[1]]
  expect_snapshot(error = TRUE, brulee:::load_tabicl_weights(model, extra))

  # Missing key.
  short <- tensors[-1]
  expect_snapshot(error = TRUE, brulee:::load_tabicl_weights(model, short))
})

test_that("tabicl_parse_config errors when fields are missing", {
  skip_on_cran()
  skip_if_not_installed("jsonlite")

  path <- withr::local_tempfile(fileext = ".json")
  jsonlite::write_json(list(embed_dim = 16), path, auto_unbox = TRUE)
  expect_snapshot(error = TRUE, brulee:::tabicl_parse_config(path))
})
