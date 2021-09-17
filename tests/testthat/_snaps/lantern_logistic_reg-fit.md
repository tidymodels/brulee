# logistic regression

    Code
      set.seed(1)
      fit <- lantern_logistic_reg(y ~ ., df, epochs = 2, verbose = TRUE)
    Message <message>
      epoch: 1 	Loss: 0.076951 
      epoch: 2 	Loss: 0.076951  x 

---

    Code
      fit
    Output
      Logistic regression
      
      100 samples, 2 features, 2 classes 
      class weights a=1, b=1 
      weight decay: 0.001 
      batch size: 90 
      validation loss after 1 epochs: 0.0769512 

# class weights - logistic regression

    Code
      set.seed(1)
      fit_imbal <- lantern_logistic_reg(y ~ ., df_imbal, verbose = TRUE,
      class_weights = 20, optimizer = "SGD")
    Message <message>
      epoch:  1 	Loss: 0.32701 
      epoch:  2 	Loss: 0.29305 
      epoch:  3 	Loss: 0.2633 
      epoch:  4 	Loss: 0.23815 
      epoch:  5 	Loss: 0.21764 
      epoch:  6 	Loss: 0.20142 
      epoch:  7 	Loss: 0.18871 
      epoch:  8 	Loss: 0.1786 
      epoch:  9 	Loss: 0.17029 
      epoch: 10 	Loss: 0.16321 
      epoch: 11 	Loss: 0.15703 
      epoch: 12 	Loss: 0.15157 
      epoch: 13 	Loss: 0.14671 
      epoch: 14 	Loss: 0.14234 
      epoch: 15 	Loss: 0.13842 
      epoch: 16 	Loss: 0.13486 
      epoch: 17 	Loss: 0.13161 
      epoch: 18 	Loss: 0.12864 
      epoch: 19 	Loss: 0.12591 
      epoch: 20 	Loss: 0.12337 

---

    Code
      set.seed(1)
      fit <- lantern_logistic_reg(y ~ ., df_imbal, epochs = 2, verbose = TRUE,
      class_weights = c(a = 12, b = 1))
    Warning <warning>
      Current loss in NaN. Training wil be stopped.

