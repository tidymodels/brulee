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
