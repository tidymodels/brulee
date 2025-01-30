# bad args

    Code
      brulee_mlp(reg_x_mat, reg_y, epochs = NA)
    Condition
      Error in `check_integer()`:
      ! brulee_mlp() expected 'epochs' to be integer.

---

    Code
      brulee_mlp(reg_x_mat, reg_y, epochs = 1:2)
    Condition
      Error in `check_integer()`:
      ! brulee_mlp() expected 'epochs' to be a single integer.

---

    Code
      brulee_mlp(reg_x_mat, reg_y, epochs = 0L)
    Condition
      Error in `check_integer()`:
      ! brulee_mlp() expected 'epochs' to be an integer on [1, Inf].

---

    Code
      brulee_mlp(reg_x_mat, reg_y, epochs = 2, hidden_units = NA)
    Condition
      Error in `check_integer()`:
      ! brulee_mlp() expected 'hidden_units' to be integer.

---

    Code
      brulee_mlp(reg_x_mat, reg_y, epochs = 2, hidden_units = -1L)
    Condition
      Error in `check_integer()`:
      ! brulee_mlp() expected 'hidden_units' to be an integer on [1, Inf].

---

    Code
      brulee_mlp(reg_x_mat, reg_y, epochs = 2, activation = NA)
    Condition
      Error in `brulee_mlp_bridge()`:
      ! `activation` should be one of: celu, elu, gelu, hardshrink, hardsigmoid, hardtanh, leaky_relu, linear, log_sigmoid, relu, relu6, rrelu, selu, sigmoid, silu, softplus, softshrink, softsign, tanh, and tanhshrink, not NA.

---

    Code
      brulee_mlp(reg_x_mat, reg_y, epochs = 2, penalty = NA)
    Condition
      Error in `check_double()`:
      ! brulee_mlp() expected 'penalty' to be a double.

---

    Code
      brulee_mlp(reg_x_mat, reg_y, epochs = 2, penalty = runif(2))
    Condition
      Error in `check_double()`:
      ! brulee_mlp() expected 'penalty' to be a single double.

---

    Code
      brulee_mlp(reg_x_mat, reg_y, epochs = 2, penalty = -1.1)
    Condition
      Error in `check_double()`:
      ! brulee_mlp() expected 'penalty' to be a double on [0, Inf].

---

    Code
      brulee_mlp(reg_x_mat, reg_y, epochs = 2, dropout = NA)
    Condition
      Error in `check_double()`:
      ! brulee_mlp() expected 'dropout' to be a double.

---

    Code
      brulee_mlp(reg_x_mat, reg_y, epochs = 2, dropout = runif(2))
    Condition
      Error in `check_double()`:
      ! brulee_mlp() expected 'dropout' to be a single double.

---

    Code
      brulee_mlp(reg_x_mat, reg_y, epochs = 2, dropout = -1.1)
    Condition
      Error in `check_double()`:
      ! brulee_mlp() expected 'dropout' to be a double on [0, 1).

---

    Code
      brulee_mlp(reg_x_mat, reg_y, epochs = 2, dropout = 1)
    Condition
      Error in `check_double()`:
      ! brulee_mlp() expected 'dropout' to be a double on [0, 1).

---

    Code
      brulee_mlp(reg_x_mat, reg_y, epochs = 2, validation = NA)
    Condition
      Error in `check_double()`:
      ! brulee_mlp() expected 'validation' to be a double.

---

    Code
      brulee_mlp(reg_x_mat, reg_y, epochs = 2, validation = runif(2))
    Condition
      Error in `check_double()`:
      ! brulee_mlp() expected 'validation' to be a single double.

---

    Code
      brulee_mlp(reg_x_mat, reg_y, epochs = 2, validation = -1.1)
    Condition
      Error in `check_double()`:
      ! brulee_mlp() expected 'validation' to be a double on [0, 1).

---

    Code
      brulee_mlp(reg_x_mat, reg_y, epochs = 2, validation = 1)
    Condition
      Error in `check_double()`:
      ! brulee_mlp() expected 'validation' to be a double on [0, 1).

---

    Code
      brulee_mlp(reg_x_mat, reg_y, epochs = 2, learn_rate = NA)
    Condition
      Error in `check_double()`:
      ! brulee_mlp() expected 'learn_rate' to be a double.

---

    Code
      brulee_mlp(reg_x_mat, reg_y, epochs = 2, learn_rate = runif(2))
    Condition
      Error in `check_double()`:
      ! brulee_mlp() expected 'learn_rate' to be a single double.

---

    Code
      brulee_mlp(reg_x_mat, reg_y, epochs = 2, learn_rate = -1.1)
    Condition
      Error in `check_double()`:
      ! brulee_mlp() expected 'learn_rate' to be a double on (0, Inf].

---

    Code
      brulee_mlp(reg_x_mat, reg_y, epochs = 2, verbose = 2)
    Condition
      Error in `check_logical()`:
      ! brulee_mlp() expected 'verbose' to be logical.

---

    Code
      brulee_mlp(reg_x_mat, reg_y, epochs = 2, verbose = rep(TRUE, 10))
    Condition
      Error in `check_logical()`:
      ! brulee_mlp() expected 'verbose' to be a single logical.

---

    Code
      brulee:::new_brulee_mlp(model_obj = bad_models$model_obj, estimates = bad_models$
        estimates, best_epoch = bad_models$best_epoch, loss = bad_models$loss, dims = bad_models$
        dims, y_stats = bad_models$y_stats, parameters = bad_models$parameters,
      blueprint = bad_models$blueprint)
    Condition
      Error in `brulee:::new_brulee_mlp()`:
      ! 'model_obj' should be a raw vector.

---

    Code
      brulee:::new_brulee_mlp(model_obj = bad_est$model_obj, estimates = bad_est$
        estimates, best_epoch = bad_est$best_epoch, loss = bad_est$loss, dims = bad_est$
        dims, y_stats = bad_est$y_stats, parameters = bad_est$parameters, blueprint = bad_est$
        blueprint)
    Condition
      Error in `brulee:::new_brulee_mlp()`:
      ! 'parameters' should be a list

---

    Code
      brulee:::new_brulee_mlp(model_obj = bad_loss$model_obj, estimates = bad_loss$
        estimates, best_epoch = bad_loss$best_epoch, loss = bad_loss$loss, dims = bad_loss$
        dims, y_stats = bad_loss$y_stats, parameters = bad_loss$parameters,
      blueprint = bad_loss$blueprint)
    Condition
      Error in `brulee:::new_brulee_mlp()`:
      ! 'loss' should be a numeric vector

---

    Code
      brulee:::new_brulee_mlp(model_obj = bad_dims$model_obj, estimates = bad_dims$
        estimates, best_epoch = bad_dims$best_epoch, loss = bad_dims$loss, dims = bad_dims$
        dims, y_stats = bad_dims$y_stats, parameters = bad_dims$parameters,
      blueprint = bad_dims$blueprint)
    Condition
      Error in `brulee:::new_brulee_mlp()`:
      ! 'dims' should be a list

---

    Code
      brulee:::new_brulee_mlp(model_obj = bad_parameters$model_obj, estimates = bad_parameters$
        estimates, best_epoch = bad_parameters$best_epoch, loss = bad_parameters$loss,
      dims = bad_parameters$dims, y_stats = bad_parameters$y_stats, parameters = bad_parameters$
        parameters, blueprint = bad_parameters$blueprint)
    Condition
      Error in `brulee:::new_brulee_mlp()`:
      ! 'dims' should be a list

---

    Code
      brulee:::new_brulee_mlp(model_obj = bad_blueprint$model_obj, estimates = bad_blueprint$
        estimates, best_epoch = bad_blueprint$best_epoch, loss = bad_blueprint$loss,
      dims = bad_blueprint$dims, y_stats = bad_blueprint$y_stats, parameters = bad_blueprint$
        parameters, blueprint = bad_blueprint$blueprint)
    Condition
      Error in `brulee:::new_brulee_mlp()`:
      ! 'blueprint' should be a hardhat blueprint

# variable hidden_units length

    Code
      model <- brulee_mlp(x, y, hidden_units = c(2, 3, 4), epochs = 1, activation = c(
        "relu", "tanh"))
    Condition
      Error in `brulee_mlp_bridge()`:
      ! 'activation' must be a single value or a vector with the same length as 'hidden_units'

---

    Code
      model <- brulee_mlp(x, y, hidden_units = c(1), epochs = 1, activation = c(
        "relu", "tanh"))
    Condition
      Error in `brulee_mlp_bridge()`:
      ! 'activation' must be a single value or a vector with the same length as 'hidden_units'

