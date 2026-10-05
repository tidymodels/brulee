# unknown checkpoint formats are rejected

    Code
      tabpfn_read_checkpoint("model.bin")
    Condition
      Error:
      ! Unknown checkpoint format for 'model.bin'.

# weight loading is strict

    Code
      tabpfn_load_state(model, missing)
    Condition
      Error:
      ! The checkpoint has no weights for 1 model parameter.
      i First missing: "layer.bias".

---

    Code
      tabpfn_load_state(model, extra)
    Condition
      Error:
      ! The checkpoint has 1 weight that the model doesn't use.
      i First unused: "other.weight".

---

    Code
      tabpfn_load_state(model, wrong_shape)
    Condition
      Error:
      ! 1 checkpoint weight has the wrong shape.
      i "layer.weight": expected 3 and 2, got 2 and 3.

# unsupported storage types are rejected

    Code
      tabpfn_read_ckpt(path)
    Condition
      Error:
      ! Can't read '<tempfile>.ckpt': the object type "torch.LongStorage" is not supported.
      i brulee reads only the checkpoint format used by the TabPFN v3 releases.

# files that aren't torch zip checkpoints are rejected

    Code
      tabpfn_read_ckpt(path)
    Condition
      Error:
      ! '<tempfile>.zip' is not a PyTorch zip checkpoint.

# the model's architecture must match the registry

    Code
      tabpfn_build_model(checkpoint, "tabpfn_v3")
    Condition
      Error:
      ! The checkpoint uses the "tabpfn_v3_5" architecture, but the registry lists "tabpfn_v3".

---

    Code
      tabpfn_build_model(checkpoint)
    Condition
      Error:
      ! The TabPFN architecture "tabpfn_v9" is not supported by brulee.
      i The checkpoint may be newer than this version of brulee.

