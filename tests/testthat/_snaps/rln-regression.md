# predict call threading surfaces predict() not the bridge

    Code
      pred <- predict(fit, x, epoch = 9999)
    Condition
      Warning in `predict()`:
      The model was fit for 3 epochs; the last epoch is used instead of epoch 9999.

