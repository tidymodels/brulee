# autoint hidden layer validation errors

    Code
      brulee_auto_int(y ~ ., data = df, hidden_units = 16L)
    Condition
      Error in `brulee_auto_int()`:
      ! Both `hidden_units` and `hidden_activations` must be provided together or both be `NULL`.

---

    Code
      brulee_auto_int(y ~ ., data = df, hidden_activations = "relu")
    Condition
      Error in `brulee_auto_int()`:
      ! Both `hidden_units` and `hidden_activations` must be provided together or both be `NULL`.

---

    Code
      brulee_auto_int(y ~ ., data = df, hidden_units = c(16L, 8L),
      hidden_activations = c("relu", "bad_name"))
    Condition
      Error in `brulee_auto_int()`:
      ! `hidden_activations` should be one of: "celu", "elu", "gelu", "hardshrink", "hardsigmoid", "hardtanh", "leaky_relu", "linear", "log_sigmoid", "relu", "relu6", "rrelu", "selu", "sigmoid", "silu", "softplus", "softshrink", "softsign", "tanh", and "tanhshrink", not "bad_name".

---

    Code
      brulee_auto_int(y ~ ., data = df, hidden_units = c(16L, 8L),
      hidden_activations = c("relu", "tanh", "elu"))
    Condition
      Error in `brulee_auto_int()`:
      ! `hidden_activations` must be a single value or a vector with the same length as `hidden_units`.

# autoint attention parameter validation errors

    Code
      brulee_auto_int(y ~ ., data = df, num_embedding = -1)
    Condition
      Error in `brulee_auto_int()`:
      ! `num_embedding` must be an integer on [1, Inf].

---

    Code
      brulee_auto_int(y ~ ., data = df, num_attn_feat = 0)
    Condition
      Error in `brulee_auto_int()`:
      ! `num_attn_feat` must be an integer on [1, Inf].

---

    Code
      brulee_auto_int(y ~ ., data = df, dropout_attn = 1.5)
    Condition
      Error in `brulee_auto_int()`:
      ! `dropout_attn` must be less than 1.

---

    Code
      brulee_auto_int(y ~ ., data = df, dropout_embedding = 1)
    Condition
      Error in `brulee_auto_int()`:
      ! `dropout_embedding` must be less than 1.

---

    Code
      brulee_auto_int(y ~ ., data = df, activation = "not_real")
    Condition
      Error in `brulee_auto_int()`:
      ! `activation` should be one of: "celu", "elu", "gelu", "hardshrink", "hardsigmoid", "hardtanh", "leaky_relu", "linear", "log_sigmoid", "relu", "relu6", "rrelu", "selu", "sigmoid", "silu", "softplus", "softshrink", "softsign", "tanh", and "tanhshrink".

# autoint default method errors on unsupported types

    Code
      brulee_auto_int(list(a = 1))
    Condition
      Error:
      ! `brulee_auto_int()` is not defined for a 'list'.

