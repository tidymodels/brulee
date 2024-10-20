# bad args

    Code
      brulee_mlp(reg_x_mat, reg_y, epochs = NA)
    Condition
      Error in `brulee_mlp()`:
      ! `epochs` must be a whole number, not `NA`.

---

    Code
      brulee_mlp(reg_x_mat, reg_y, epochs = 1:2)
    Condition
      Error in `brulee_mlp()`:
      ! `epochs` must be a whole number, not an integer vector.

---

    Code
      brulee_mlp(reg_x_mat, reg_y, epochs = 0L)
    Condition
      Error in `brulee_mlp()`:
      ! `epochs` must be a whole number larger than or equal to 1, not the number 0.

---

    Code
      brulee_mlp(reg_x_mat, reg_y, epochs = 2, hidden_units = NA)
    Condition
      Error in `brulee_mlp()`:
      ! `hidden_units` must be a whole number, not `NA`.

---

    Code
      brulee_mlp(reg_x_mat, reg_y, epochs = 2, hidden_units = -1L)
    Condition
      Error in `brulee_mlp()`:
      ! `hidden_units` must be a whole number larger than or equal to 1, not the number -1.

---

    Code
      brulee_mlp(reg_x_mat, reg_y, epochs = 2, activation = NA)
    Condition
      Error in `brulee_mlp()`:
      ! `activation` must be a character vector, not `NA`.

---

    Code
      brulee_mlp(reg_x_mat, reg_y, epochs = 2, penalty = NA)
    Condition
      Error in `brulee_mlp()`:
      ! `penalty` must be a number, not `NA`.

---

    Code
      brulee_mlp(reg_x_mat, reg_y, epochs = 2, penalty = runif(2))
    Condition
      Error in `brulee_mlp()`:
      ! `penalty` must be a number, not a double vector.

---

    Code
      brulee_mlp(reg_x_mat, reg_y, epochs = 2, penalty = -1.1)
    Condition
      Error in `brulee_mlp()`:
      ! `penalty` must be a number larger than or equal to 0, not the number -1.1.

---

    Code
      brulee_mlp(reg_x_mat, reg_y, epochs = 2, dropout = NA)
    Condition
      Error in `brulee_mlp()`:
      ! `dropout` must be a number, not `NA`.

---

    Code
      brulee_mlp(reg_x_mat, reg_y, epochs = 2, dropout = runif(2))
    Condition
      Error in `brulee_mlp()`:
      ! `dropout` must be a number, not a double vector.

---

    Code
      brulee_mlp(reg_x_mat, reg_y, epochs = 2, dropout = -1.1)
    Condition
      Error in `brulee_mlp()`:
      ! `dropout` must be a number between 0 and 1, not the number -1.1.

---

    Code
      brulee_mlp(reg_x_mat, reg_y, epochs = 2, dropout = 1)
    Condition
      Error in `brulee_mlp()`:
      ! `dropout` must be a number between 0 and 1, not the number 1.

---

    Code
      brulee_mlp(reg_x_mat, reg_y, epochs = 2, validation = NA)
    Condition
      Error in `brulee_mlp()`:
      ! `validation` must be a number, not `NA`.

---

    Code
      brulee_mlp(reg_x_mat, reg_y, epochs = 2, validation = runif(2))
    Condition
      Error in `brulee_mlp()`:
      ! `validation` must be a number, not a double vector.

---

    Code
      brulee_mlp(reg_x_mat, reg_y, epochs = 2, validation = -1.1)
    Condition
      Error in `brulee_mlp()`:
      ! `validation` must be a number between 0 and 1, not the number -1.1.

---

    Code
      brulee_mlp(reg_x_mat, reg_y, epochs = 2, validation = 1)
    Condition
      Error in `brulee_mlp()`:
      ! `validation` must be a number between 0 and 1, not the number 1.

---

    Code
      brulee_mlp(reg_x_mat, reg_y, epochs = 2, learn_rate = NA)
    Condition
      Error in `brulee_mlp()`:
      ! `learn_rate` must be a number, not `NA`.

---

    Code
      brulee_mlp(reg_x_mat, reg_y, epochs = 2, learn_rate = runif(2))
    Condition
      Error in `brulee_mlp()`:
      ! `learn_rate` must be a number, not a double vector.

---

    Code
      brulee_mlp(reg_x_mat, reg_y, epochs = 2, learn_rate = -1.1)
    Condition
      Error in `brulee_mlp()`:
      ! `learn_rate` must be a number larger than or equal to 2.22044604925031e-16, not the number -1.1.

---

    Code
      brulee_mlp(reg_x_mat, reg_y, epochs = 2, verbose = 2)
    Condition
      Error in `brulee_mlp()`:
      ! `verbose` should be a single logical value, not the number 2.

---

    Code
      brulee_mlp(reg_x_mat, reg_y, epochs = 2, verbose = rep(TRUE, 10))
    Condition
      Error in `brulee_mlp()`:
      ! `verbose` should be a single logical value, not a logical vector.

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
      Error in `brulee_mlp()`:
      ! 2 activations were given for 3 hidden layers.

---

    Code
      model <- brulee_mlp(x, y, hidden_units = c(1), epochs = 1, activation = c(
        "relu", "tanh"))
    Condition
      Error in `brulee_mlp()`:
      ! 2 activations were given for 1 hidden layer.

