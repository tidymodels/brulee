lantern_mn_reg_fit_imp <-
  function(x, y,
           epochs = 10L,
           batch_size = NULL,
           learn_rate = 0.01,
           penalty = 0,
           loss_function = "mse",
           verbose = FALSE,
           ...) {

    ## ---------------------------------------------------------------------------
    f_nm <- "lantern_linear_reg"
    # check values of various argument values
    check_integer(epochs, single = TRUE, 2, fn = f_nm)
    check_double(learn_rate, single = TRUE, 0, incl = c(FALSE, TRUE), fn = f_nm)
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
    if (!is.factor(y) || length(levels(y)) != 2) {
      rlang::abort("'y' should be a factor vector.")
    }
    lvls <- levels(y)

    ## ---------------------------------------------------------------------------
    # Check missing values
    compl_data <- check_missing_data(x, y, "lantern_linear_reg", verbose)
    x <- compl_data$x
    y <- compl_data$y

    # Set and check batch size
    if (is.null(batch_size)) {
      batch_size <- nrow(x)
    } else {
      batch_size <- min(batch_size, nrow(x))
    }
    check_integer(batch_size, single = TRUE, 2, nrow(x), incl = c(TRUE, TRUE), fn = f_nm)

    ## ---------------------------------------------------------------------------
    # Convert to index sampler and data loader
    ds <- matrix_to_dataset(x, y)
    dl <- dataloader(ds, batch_size = batch_size, shuffle = TRUE)

    ## ---------------------------------------------------------------------------
    # Initialize model and optimizer
    model <- multinomial_reg_module(ncol(x), length(lvls))
    # Write a optim wrapper
    optimizer <- torch::optim_sgd(model$parameters, lr = learn_rate)

    ## ---------------------------------------------------------------------------
    # Optimize parameters
    for (epoch in 1:epochs) {
      # Over batches
      for (b in enumerate(dl)) {
        optimizer$zero_grad()
        output <- model(b[[1]])
        loss <- torch::nn_bce_loss(output, b[[2]])
        loss$backward()
        optimizer$step()
        print(loss$item())
      }
    }

    print(model$parameters)
    ## ---------------------------------------------------------------------------
    # convert results to R objects
    beta <-
      c(
        as.array(model$parameters$fc1.bias),
        as.array(model$parameters$fc1.weight)[1,]
      )
    names(beta) <- c("(Intercept)", colnames(x))

    list(coefficients = beta, loss = numeric(0))
  }

## -----------------------------------------------------------------------------
multinomial_reg_module <-
  torch::nn_module(
    "multinomia_reg",
    initialize = function(num_pred, num_class) {
      self$fc1 <- torch::nn_linear(num_pred, 1)
    },
    forward = function(x) {
      lp_to_prob <- nn_sigmoid()
      # returns a single probability value from the linear predictor
      x %>% self$fc1() %>% lp_to_prob()
    }
  )
