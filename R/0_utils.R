
# used for autoplots
lantern_plot <- function(object, ...) {
 x <- tibble::tibble(iteration = seq(along = object$loss), loss = object$loss)

 if(object$parameters$validation > 0) {
  if (is.na(object$y_stats$mean)) {
   lab <- "\n(validation set)"
  } else {
   lab <- "\n(validation set, scaled)"
  }
 } else {
  if (is.na(object$y_stats$mean)) {
   lab <- "\n(training set)"
  } else {
   lab <- "\n(training set, scaled)"
  }
 }
 lab <- paste(object$loss_type, lab)

 ggplot2::ggplot(x, ggplot2::aes(x = iteration, y = loss)) +
  ggplot2::geom_line() +
  ggplot2::labs(y = lab)+
  ggplot2::geom_vline(xintercept = object$best_epoch, lty = 2, col = "green")
}

# ------------------------------------------------------------------------------
# used in print methods

lantern_print <- function(x, ...) {
 lvl <- get_levels(x)
 if (is.null(lvl)) {
  chr_y <- "numeric outcome"
 } else {
  chr_y <- paste(length(lvl), "classes")
 }
 cat(
  format(x$dims$n, big.mark = ","), "samples,",
  format(x$dims$p, big.mark = ","), "features,",
  chr_y, "\n"
 )
 if (!is.null(x$parameters$class_weights)) {
  cat("class weights",
      paste0(
       names(x$parameters$class_weights),
       "=",
       format(x$parameters$class_weights),
       collapse = ", "
      ),
      "\n")
 }
 if (x$parameters$penalty > 0) {
  cat("weight decay:", x$parameters$penalty, "\n")
 }
 if (any(names(x$parameters) == "dropout")) {
  cat("dropout proportion:", x$parameters$dropout, "\n")
 }
 cat("batch size:", x$parameters$batch_size, "\n")

 if (!is.null(x$loss)) {
  it <- x$best_epoch
  if(x$parameters$validation > 0) {
   if (is.na(x$y_stats$mean)) {
    cat("validation", x$loss_type, "after", it, "epochs:",
        signif(x$loss[it], 3), "\n")
   } else {
    cat("scaled validation", x$loss_type, "after", it, "epochs:",
        signif(x$loss[it], 3), "\n")
   }
  } else {
   if (is.na(x$y_stats$mean)) {
    cat("training set", x$loss_type, "after", it, "epochs:",
        signif(x$loss[it], 3), "\n")
   } else {
    cat("scaled training set", x$loss_type, "after", it, "epochs:",
        signif(x$loss[it], 3), "\n")
   }
  }
 }
 invisible(x)
}

# ------------------------------------------------------------------------------

lantern_coefs <- function(object, epoch = NULL, ...) {
 if (is.null(epoch)) {
  epoch <- object$best_epoch
 }
 module <- revive_model(object, epoch = epoch)
 parameters <- module$parameters
 lapply(parameters, as.array)
}

# ------------------------------------------------------------------------------


model_to_raw <- function(model) {
 con <- rawConnection(raw(), open = "w")
 on.exit({close(con)}, add = TRUE)
 torch::torch_save(model, con)
 r <- rawConnectionValue(con)
 r
}

# ------------------------------------------------------------------------------

reg_loss_fn <- function(input, target, loss) {
  loss_fn <- get_loss_fn(loss)
  loss_fn(input, target$view(c(-1,1)))
}


cls_loss_fn <- function(input, target, weights, loss) {
  loss_fn <- get_loss_fn(loss)
  loss_fn(
    weight = weights,
    input = torch::torch_log(input),
    target = target
  )
}

get_loss_fn <- function(x) {
  switch(x,
         mse = torch::nnf_mse_loss,
         mae = torch::nnf_l1_loss,
         poisson = torch::nn_poisson_nll_loss,
         'log_loss' = torch::nn_nll_loss,
         "cross_entropy" = nn_cross_entropy_loss,
  )
}
