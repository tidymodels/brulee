# logistic regression

    Code
      set.seed(1)
      fit <- lantern_logistic_reg(y ~ ., df, epochs = 2, verbose = TRUE)
    Message <message>
      epoch: 1 	Loss: 0.077 
      epoch: 2 	Loss: 0.077  x 

---

    Code
      fit
    Output
      Logistic regression
      
      100 samples, 2 features, 2 classes 
      class weights a=1, b=1 
      weight decay: 0.001 
      batch size: 90 
      validation loss after 1 epochs: 0.077 

# class weights - logistic regression

    Code
      set.seed(1)
      fit_imbal <- lantern_logistic_reg(y ~ ., df_imbal, verbose = TRUE,
      class_weights = 20, optimizer = "SGD")
    Message <message>
      epoch:  1 	Loss: 0.327 
      epoch:  2 	Loss: 0.293 
      epoch:  3 	Loss: 0.263 
      epoch:  4 	Loss: 0.238 
      epoch:  5 	Loss: 0.218 
      epoch:  6 	Loss: 0.201 
      epoch:  7 	Loss: 0.189 
      epoch:  8 	Loss: 0.179 
      epoch:  9 	Loss: 0.17 
      epoch: 10 	Loss: 0.163 
      epoch: 11 	Loss: 0.157 
      epoch: 12 	Loss: 0.152 
      epoch: 13 	Loss: 0.147 
      epoch: 14 	Loss: 0.142 
      epoch: 15 	Loss: 0.138 
      epoch: 16 	Loss: 0.135 
      epoch: 17 	Loss: 0.132 
      epoch: 18 	Loss: 0.129 
      epoch: 19 	Loss: 0.126 
      epoch: 20 	Loss: 0.123 

---

    Code
      set.seed(1)
      fit <- lantern_logistic_reg(y ~ ., df_imbal, epochs = 2, verbose = TRUE,
      class_weights = c(a = 12, b = 1))
    Warning <warning>
      Current loss in NaN. Training wil be stopped.

