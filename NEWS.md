# brulee 0.7.0

New models for tabular data:

  * Regularization Learning Networks (`brulee_rln()`) use a conventional MLP architecture but each weight learns its own adaptive regularization coefficient.
  * ResNet (`brulee_resnet()`) can fit a multilayer neural network with skip (i.e. residual) connections and batch normalization. 
  * AutoInt (`brulee_auto_int()`) uses residual connections and columnwise attention mechanisms to create embeddings that encourage in-context learning of features. 
  * Saint (`brulee_saint()`) uses column and/or row attention mechanisms. 

* All modeling functions now support GPU acceleration via the `device` parameter. Users can specify `device = "cpu"`, `device = "cuda"`, or `device = "mps"` (Apple Silicon). When `device = NULL` (default), the package automatically selects CUDA if available, otherwise defaults to CPU. Note: MPS is not auto-selected because it doesn't support float64 dtype required by brulee. See`?training_efficiency` for some related notes. 

## Breaking Changes

* Float tensors were changed from 64-bit floats to 32-bit. This is to enable GPU usage on MPS devices. 

* Parameters are initialized on CPU devices and then converted to the chosen device. In some cases, the RNG initialization code is independent of the seed. 

* For classification, the softmax was moved out of every model's forward pass so the loss can use `torch::nnf_cross_entropy()` (which applies the log-sum-exp trick internally) instead of `nll_loss(log(softmax(x)))`. This avoids `log(0)` underflow that produced `NaN` losses and "numerical overflow" early stopping on overspecified `brulee_saint()` / `brulee_auto_int()` fits. Affects `brulee_mlp()`, `brulee_logistic_reg()`, `brulee_multinomial_reg()`, `brulee_resnet()`, `brulee_auto_int()`, and `brulee_saint()`. New fits carry `output_type = "logits"` so the predict path applies softmax; serialized fits from earlier versions of brulee continue to predict correctly.

# brulee 0.6.0

* Transition from the magrittr pipe to the base R pipe.

* To try to help avoiding numeric overflow in the loss functions: 

  * Tensors are stored as a 64-bit float instead of 32-bit. 
  
  * Starting values were transitioned to using Gaussian distribution (instead of uniform) with a smaller standard deviation. 
  
  * The results always contain the initial results to use as a fallback if there is overflow during the first epoch.

  * `brulee_mlp()` has two additional parameters, `grad_value_clip` and `grad_value_clip`, that prevent issues. 
  
  * The warning was changed to "Early stopping occurred at epoch {X} due to numerical overflow of the loss function."

* Several new SGD optimizers were added: `"ADAMw"`, `"Adadelta"`, `"Adagrad"`, and `"RMSprop"`.

* Mixture parameter values different than zero cannot be used for several optimizers since they require L2 penalties. 

# brulee 0.5.0

 * Removed a unit test for numerical overflow since it occurs less frequently and has become increasingly more challenging to reproduce.
 
# brulee 0.4.0

* Added a convenience function, `brulee_mlp_two_layer()`, to more easily fit two-layer networks with parsnip.  

* Various changes and improvements to error and warning messages. 

* Fixed a bug that occurred when linear activation was used for neural networks (#68). 

# brulee 0.3.0

* Fixed bug where `coef()` didn't would error if used on a `brulee_logistic_reg()` that was trained with a recipe. (#66)

* Fixed a bug where SGD always being used as the optimizer (#61). 

* Additional activation functions were added (#74). 

# brulee 0.2.0

* Several learning rate schedulers were added to the modeling functions (#12).

* An `optimizer` was added to [brulee_mlp()], with a new default being LBFGS instead of stochastic gradient descent. 

# brulee 0.1.0

* Modeling functions gained a `mixture` argument for the proportion of L1 penalty that is used. (#50)

* Penalization was not occurring when quasi-Newton optimization was chosen. (#50)

# brulee 0.0.1

First CRAN release.
