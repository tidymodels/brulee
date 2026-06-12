# Training Efficiency

There are ways to speed up or slow down model training. Here are some
notes.

## Details

GPUs can perform calculations very fast, sometimes faster than the
overhead of a high-level interface such as brulee. GPU utilization might
be lower than expected because the model is not very large (i.e., with
millions of parameters) and/or because the batch size is small.

For the latter, here is an example of a training set with 1K samples,
one single hidden layer with 50 units, 200 epochs, and used ADAMw
optimizer:

                                             (CPU/CUDA)
    batch_size   CPU elapsed   CUDA elapsed     speedup
           128        90.09s        111.54s       0.81x
           512        26.22s         28.61s       0.92x
          2048        11.07s          8.31s       1.33x
          8192         4.42s          3.57s       1.24x

As batch sizes become larger, the GPU has a better chance of reducing
training time.

Some optimizers are faster than others. Although we use
[`torch::optim_adamw()`](https://torch.mlverse.org/docs/reference/optim_adamw.html)
directly, it can be much slower than others. For one benchmark:

    optimizer    CPU elapsed   CUDA elapsed
        ADAMw         66.22s         84.42s
          SGD         30.12s         30.83s
