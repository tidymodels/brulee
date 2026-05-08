#' Chronos-2 pretrained forecasting model
#'
#' `brulee_chronos()` loads a pretrained Chronos-2 time series forecasting
#' quantile regresison model from HuggingFace. Unlike other brulee models, no
#' training is performed. The model is a foundation model with fixed pretrained
#' weights.
#'
#' @param model_id A character string identifying the HuggingFace model
#'   repository to download. Default: `"amazon/chronos-2"` (120M parameters).
#' @param prediction_length An integer for the number of future time steps to
#'   forecast. Default: `NULL` (uses the model maximum of 1024). Must not
#'   exceed the model maximum.
#' @param quantile_levels A numeric vector of quantile levels to produce in
#'   predictions. Must be a subset of the model's trained quantiles. Default:
#'   `(1:9) / 10`.
#' @param device A character string for the computation device: `"cpu"`,
#'   `"cuda"`, or `"mps"`. Default: `NULL` (auto-detects best available).
#' @param cache_dir Path to a directory for caching downloaded model files.
#'   Default: `"~/.cache/chronos-r"`.
#'
#' @returns A `brulee_chronos` object with elements:
#'
#'   * `model`: The torch `nn_module` (in eval mode, on the specified device).
#'   * `config`: Parsed model configuration list.
#'   * `device`: The torch device in use.
#'   * `prediction_length`: Validated prediction length.
#'   * `quantile_levels`: Validated quantile levels.
#'
#' @examples
#' \dontrun{
#' mod <- brulee_chronos()
#' predict(mod, data.frame(
#'   item_id = "air_passengers",
#'   timestamp = seq(as.Date("1949-01-01"), by = "month", length.out = 144),
#'   target = as.numeric(AirPassengers)
#' ))
#' }
#' @export
brulee_chronos <- function(
  model_id = "amazon/chronos-2",
  prediction_length = NULL,
  quantile_levels = (1:9) / 10,
  device = NULL,
  cache_dir = file.path(Sys.getenv("HOME"), ".cache", "chronos-r")
) {
 # TODO:
 # - make control file for model id and cache dir
 # Resolve device
 if (is.null(device)) {
  device <- chronos2_detect_device()
 } else {
  device <- torch::torch_device(device)
 }

 # Download model files
 model_dir <- chronos2_download(model_id, cache_dir = cache_dir)

 # Parse config
 config <- chronos2_parse_config(file.path(model_dir, "config.json"))

 # Validate prediction_length
 max_prediction_length <- config$max_output_patches * config$output_patch_size
 if (is.null(prediction_length)) {
  prediction_length <- max_prediction_length
 }
 if (prediction_length > max_prediction_length) {
  cli::cli_abort(
   "{.arg prediction_length} ({prediction_length}) exceeds model maximum ({max_prediction_length})."
  )
 }
 if (prediction_length < 1) {
  cli::cli_abort("{.arg prediction_length} must be at least 1.")
 }

 # Validate quantile_levels
 model_quantiles <- config$quantiles
 unavailable <- setdiff(quantile_levels, model_quantiles)
 if (length(unavailable) > 0) {
  cli::cli_abort(c(
   "Requested quantile levels not available in model: {.val {unavailable}}.",
   "i" = "Available: {.val {model_quantiles}}"
  ))
 }

 # Build model, load weights, move to device

 model <- chronos2_model(config)
 load_chronos2_weights(model, file.path(model_dir, "model.safetensors"))
 model$to(device = device)
 model$eval()

 structure(
  list(
   model = model,
   config = config,
   device = device,
   prediction_length = as.integer(prediction_length),
   quantile_levels = quantile_levels
  ),
  class = "brulee_chronos"
 )
}

#' @export
print.brulee_chronos <- function(x, ...) {
 cat(cli::style_bold("Chronos-2 Pretrained Forecasting Mode"), "\n\n", sep = "")

 mod_lst <-
  c(
   " " = "Model: {x$config$d_model}",
   " " = "Layers: { x$config$num_layers}",
   " " = "Attention heads: {x$config$num_heads}",
   " " = "Quantiles: {x$prediction_length}",
   " " = "Prediction length: {x$quantile_levels}",
   " " = "Device: {x$device}",
  )

 cli::cli_bullets(mod_lst)

 invisible(x)
}

# ─── Internal helpers ─────────────────────────────────────────────────────────

chronos2_detect_device <- function() {
 if (torch::cuda_is_available()) {
  torch::torch_device("cuda")
 } else if (torch::backends_mps_is_available()) {
  torch::torch_device("mps")
 } else {
  torch::torch_device("cpu")
 }
}

chronos2_download <- function(
  model_id = "amazon/chronos-2",
  cache_dir = file.path(Sys.getenv("HOME"), ".cache", "chronos-r")
) {
 model_dir <- file.path(cache_dir, gsub("/", "--", model_id))
 dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)

 files <- c("config.json", "model.safetensors")

 for (f in files) {
  dest <- file.path(model_dir, f)
  if (!file.exists(dest)) {
   url <- sprintf("https://huggingface.co/%s/resolve/main/%s", model_id, f)
   cli::cli_progress_step("Downloading {.url {url}}")
   download.file(url, dest, mode = "wb", quiet = TRUE)
  }
 }

 model_dir
}
