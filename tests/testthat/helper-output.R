# Capture all output (stdout + messages) from an expression and return as
# a character vector. Useful for testing print/summary methods that mix
# cat() (stdout) and cli (messages/conditions).
capture_all_output <- function(expr) {
  stdout <- capture.output(
    msgs <- capture.output(expr, type = "message")
  )
  c(stdout, msgs)
}
