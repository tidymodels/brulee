torch_linear_reg_fit_imp <-
 function(x, y,
          epochs = 100L,
          learning_rate = 0.01,
          penalty = 0,
          conv_crit = 0,
          optimizer = "SGD",
          loss_function = "mse",
          verbose = FALSE,
          ...) {

  ## ---------------------------------------------------------------------------
  f_nm <- "torch_linear_reg"
  # check values of various argument values
  check_integer(epochs, single = TRUE, 2, fn = f_nm)
  check_double(learning_rate, single = TRUE, 0, incl = c(FALSE, TRUE), fn = f_nm)
  check_double(penalty, single = TRUE, 0, incl = c(FALSE, TRUE), fn = f_nm)
  check_logical(verbose, single = TRUE, fn = f_nm)

  # check matrices/vectors, matrix type, matrix column names
  if (!is.matrix(x) || !is.numeric(x)) {
   rlang::abort("'x' should be a numeric matrix.")
  }
  nms <- colnames(x)
  if (length(nms) != ncol(x)) {
   rlang::abort("Every column of 'x' should have a name.")
  }
  if (!is.vector(y)) {
   rlang::abort("'y' should be a vector.")
  }

  ## ---------------------------------------------------------------------------
  # Check missing values
  compl_data <- check_missing_data(x, y, "torch_linear_reg", verbose)
  x <- compl_data$x
  y <- compl_data$y

  ## ---------------------------------------------------------------------------
  # Convert to index sampler and data loader
  ds <- matrix_to_dataset(x, y)
  dl <- torch::dataloader(ds)

  ## ---------------------------------------------------------------------------
  # Initialize model and optimizer
  model <- linear_reg_module(ncol(x))
  model$parameters$fc1.bias$set_data(torch::torch_tensor(mean(y)))

  # Write a optim wrapper
  optimizer <- torch::optim_sgd(model$parameters, lr = learning_rate)

  ## ---------------------------------------------------------------------------

  loss_prev <- 10^38
  loss_vec <- rep(NA_real_, epochs)
  epoch_chr <- format(1:epochs)

  # Optimize parameters
  for (epoch in 1:epochs) {

    pred <- model(dl$dataset$data$x)[,1]
    loss <- torch::nnf_mse_loss(pred, dl$dataset$data$y)

    loss_curr <- as.array(loss)
    loss_vec[epoch] <- loss_curr
    loss_diff <- (loss_prev - loss_curr)/loss_prev
    loss_prev <- loss_curr

    if (loss_diff <= conv_crit) {
      break()
    }

    if (verbose) {
      message("epoch:", epoch_chr[epoch], "\tRMSE:", signif(sqrt(loss_curr), 5))
    }

    optimizer$zero_grad()
    loss$backward()
    optimizer$step()
  }

  ## ---------------------------------------------------------------------------
  # convert results to R objects

  beta <-
   c(
    as.array(model$parameters$fc1.bias),
    as.array(model$parameters$fc1.weight)[1,]
   )
  names(beta) <- c("(Intercept)", colnames(x))

  list(coefficients = beta, loss = sqrt(loss_vec[!is.na(loss_vec)]))
 }

## -----------------------------------------------------------------------------

linear_reg_module <-
 torch::nn_module(
  "linear_reg",
  initialize = function(num_pred) {
   self$fc1 <- torch::nn_linear(num_pred, 1)
  },
  forward = function(x) {
   x %>% self$fc1()
  }
 )

