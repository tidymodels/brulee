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
