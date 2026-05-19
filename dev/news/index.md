# Changelog

## brulee (development version)

- Regularization Learning Networks (RLN) can now be fit via
  [`brulee_rln()`](https://brulee.tidymodels.org/dev/reference/brulee_rln.md)

- ResNet models can now be fit via
  [`brulee_resnet()`](https://brulee.tidymodels.org/dev/reference/brulee_resnet.md).

## brulee 0.6.0

CRAN release: 2025-09-02

- Transition from the magrittr pipe to the base R pipe.

- To try to help avoiding numeric overflow in the loss functions:

  - Tensors are stored as a 64-bit float instead of 32-bit.

  - Starting values were transitioned to using Gaussian distribution
    (instead of uniform) with a smaller standard deviation.

  - The results always contain the initial results to use as a fallback
    if there is overflow during the first epoch.

  - [`brulee_mlp()`](https://brulee.tidymodels.org/dev/reference/brulee_mlp.md)
    has two additional parameters, `grad_value_clip` and
    `grad_value_clip`, that prevent issues.

  - The warning was changed to “Early stopping occurred at epoch {X} due
    to numerical overflow of the loss function.”

- Several new SGD optimizers were added: `"ADAMw"`, `"Adadelta"`,
  `"Adagrad"`, and `"RMSprop"`.

- Mixture parameter values different than zero cannot be used for
  several optimizers since they require L2 penalties.

## brulee 0.5.0

CRAN release: 2025-04-07

- Removed a unit test for numerical overflow since it occurs less
  frequently and has become increasingly more challenging to reproduce.

## brulee 0.4.0

CRAN release: 2025-01-30

- Added a convenience function,
  [`brulee_mlp_two_layer()`](https://brulee.tidymodels.org/dev/reference/brulee_mlp.md),
  to more easily fit two-layer networks with parsnip.

- Various changes and improvements to error and warning messages.

- Fixed a bug that occurred when linear activation was used for neural
  networks ([\#68](https://github.com/tidymodels/brulee/issues/68)).

## brulee 0.3.0

CRAN release: 2024-02-14

- Fixed bug where [`coef()`](https://rdrr.io/r/stats/coef.html) didn’t
  would error if used on a
  [`brulee_logistic_reg()`](https://brulee.tidymodels.org/dev/reference/brulee_logistic_reg.md)
  that was trained with a recipe.
  ([\#66](https://github.com/tidymodels/brulee/issues/66))

- Fixed a bug where SGD always being used as the optimizer
  ([\#61](https://github.com/tidymodels/brulee/issues/61)).

- Additional activation functions were added
  ([\#74](https://github.com/tidymodels/brulee/issues/74)).

## brulee 0.2.0

CRAN release: 2022-09-19

- Several learning rate schedulers were added to the modeling functions
  ([\#12](https://github.com/tidymodels/brulee/issues/12)).

- An `optimizer` was added to \[brulee_mlp()\], with a new default being
  LBFGS instead of stochastic gradient descent.

## brulee 0.1.0

CRAN release: 2022-02-02

- Modeling functions gained a `mixture` argument for the proportion of
  L1 penalty that is used.
  ([\#50](https://github.com/tidymodels/brulee/issues/50))

- Penalization was not occurring when quasi-Newton optimization was
  chosen. ([\#50](https://github.com/tidymodels/brulee/issues/50))

## brulee 0.0.1

CRAN release: 2021-12-15

First CRAN release.
