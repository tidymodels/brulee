# tabicl_resolve_device defaults to cpu and refuses mps

    Code
      dev <- brulee:::tabicl_resolve_device("mps")
    Condition
      Warning:
      The MPS backend is not supported for TabICL; using "cpu".
      i The bundled libtorch MPS kernels crash on parts of the model.

