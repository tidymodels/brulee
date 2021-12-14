brulee_coefs <- function(object, epoch = NULL, ...) {
 if (!is.null(epoch) && length(epoch) != 1) {
  rlang::abort("'epoch' should be a single integer.")
 }
 max_epochs <- length(object$estimates)

 if (is.null(epoch)) {
  epoch <- object$best_epoch
 } else {
  if (epoch > max_epochs) {
   msg <- glue::glue("There were only {max_epochs} epochs fit. Setting 'epochs' to {max_epochs}.")
   rlang::warn(msg)
   epoch <- max_epochs
  }

 }
 object$estimates[[epoch]]
}


#' Extract Model Coefficients
#'
#' @param object A model fit from \pkg{brulee}.
#' @param epoch A single integer for the training iteration. If left `NULL`,
#' the estimates from the best model fit (via internal performance metrics).
#' @param ... Not currently used.
#' @return For logistic/linear regression, a named vector. For neural networks,
#' a list of arrays.
#' @examples
#' \donttest{
#' if (torch::torch_is_installed()) {
#'
#'  data(ames, package = "modeldata")
#'
#'  ames$Sale_Price <- log10(ames$Sale_Price)
#'
#'  set.seed(1)
#'  in_train <- sample(1:nrow(ames), 2000)
#'  ames_train <- ames[ in_train,]
#'  ames_test  <- ames[-in_train,]
#'
#'  # Using recipe
#'  library(recipes)
#'
#'  ames_rec <-
#'   recipe(Sale_Price ~ Longitude + Latitude, data = ames_train) %>%
#'     step_normalize(all_numeric_predictors())
#'
#'  set.seed(2)
#'  fit <- brulee_linear_reg(ames_rec, data = ames_train,
#'                            epochs = 50, batch_size = 32)
#'
#'  coef(fit)
#'  coef(fit, epoch = 1)
#' }
#' }
#' @name brulee-coefs
#' @export
coef.brulee_logistic_reg <- function(object, epoch = NULL, ...) {
 network_params <- brulee_coefs(object, epoch)
 slopes <- network_params$fc1.weight[,2] - network_params$fc1.weight[,1]
 int <- network_params$fc1.bias[2] - network_params$fc1.bias[1]
 param <- c(int, slopes)
 names(param) <- c("(Intercept)", object$dims$features)
 param
}

#' @rdname brulee-coefs
#' @export
coef.brulee_linear_reg <- function(object, epoch = NULL, ...) {
 network_params <- brulee_coefs(object, epoch)
 slopes <- network_params$fc1.weight[1,]
 int <- network_params$fc1.bias
 param <- c(int, slopes)
 names(param) <- c("(Intercept)", object$dims$features)
 param
}

#' @rdname brulee-coefs
#' @export
coef.brulee_mlp <- brulee_coefs

#' @rdname brulee-coefs
#' @export
coef.brulee_multinomial_reg <- function(object, epoch = NULL, ...) {
 network_params <- brulee_coefs(object, epoch)
 slopes <- t(network_params$fc1.weight)
 int <- network_params$fc1.bias
 param <- rbind(int, slopes)
 rownames(param) <- c("(Intercept)", object$dims$features)
 colnames(param) <- object$dims$levels
 param
}

