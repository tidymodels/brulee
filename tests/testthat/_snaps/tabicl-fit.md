# tabicl_resolve_device defaults to cpu and refuses mps

    Code
      dev <- brulee:::tabicl_resolve_device("mps")
    Condition
      Warning:
      The MPS backend is not supported for TabICL; using "cpu".
      i The bundled libtorch MPS kernels crash on parts of the model.

# brulee_tab_icl errors when path is missing

    Code
      brulee_tab_icl(x_train, y_train)
    Condition
      Error in `brulee_tab_icl()`:
      ! Automatic TabICL weight download is not available yet.
      i Convert a released checkpoint and pass its directory via `path`.

