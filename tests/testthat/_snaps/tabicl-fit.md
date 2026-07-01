# tabicl_resolve_device defaults to cpu and refuses mps

    Code
      dev <- brulee:::tabicl_resolve_device("mps")
    Condition
      Warning:
      The MPS backend is not supported for TabICL; using "cpu".
      i The bundled libtorch MPS kernels crash on parts of the model.

# tabicl_subsample_indices errors when limit < number of classes

    Code
      brulee:::tabicl_subsample_indices(outcome, 3)
    Condition
      Error:
      ! `training_set_limit` (3) is smaller than the number of outcome classes (4); cannot keep at least one row per class.

