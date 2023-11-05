#' Determine an appropriate computational device for torch
#'
#' Uses \pkg{torch} functions to determine if there is a GPU available for use.
#' @return A character string, one of: `"cpu"`, `"cuda"`, or `"mps"`.
#' @examples
#' guess_brulee_device()
#' @export
guess_brulee_device <- function() {
 if (torch::backends_mps_is_available()) {
  dev <- "mps"
 } else if (torch::cuda_is_available()) {
  dev <- "cuda"
 } else {
  dev <- "cpu"
 }
 dev
}
