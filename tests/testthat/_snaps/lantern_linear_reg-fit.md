# linear regression test

    Code
      set.seed(1)
      fit <- lantern_linear_reg(y ~ ., df, epochs = 2, verbose = TRUE)
    Message <message>
      epoch: 1 	Loss (scaled): 1.46e-12 
      epoch: 2 	Loss (scaled): 1.46e-12  x 
    Code
      fit
    Output
      Linear regression
      
      100 samples, 2 features, numeric outcome 
      weight decay: 0.001 
      batch size: 90 
      scaled validation loss after 1 epochs: 1.46e-12 

