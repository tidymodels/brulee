# logistic regression

    Code
      set.seed(1)
      fit <- brulee_logistic_reg(y ~ ., df, epochs = 2, verbose = TRUE, penalty = 0)
    Message <rlang_message>
      epoch: 1 	Loss: 0.173 
      epoch: 2 	Loss: 0.173  x 

---

    Code
      fit
    Output
      Logistic regression
      
      1,000 samples, 2 features, 2 classes 
      class weights a=1, b=1 
      batch size: 900 
      validation loss after 1 epoch: 0.173 

# class weights - logistic regression

    Code
      set.seed(1)
      fit_imbal <- brulee_logistic_reg(y ~ ., df_imbal, verbose = TRUE,
      class_weights = 20, optimizer = "SGD", penalty = 0)
    Message <rlang_message>
      epoch:  1 	Loss: 0.329 
      epoch:  2 	Loss: 0.302 
      epoch:  3 	Loss: 0.282 
      epoch:  4 	Loss: 0.267 
      epoch:  5 	Loss: 0.255 
      epoch:  6 	Loss: 0.245 
      epoch:  7 	Loss: 0.236 
      epoch:  8 	Loss: 0.228 
      epoch:  9 	Loss: 0.222 
      epoch: 10 	Loss: 0.216 
      epoch: 11 	Loss: 0.211 
      epoch: 12 	Loss: 0.206 
      epoch: 13 	Loss: 0.202 
      epoch: 14 	Loss: 0.198 
      epoch: 15 	Loss: 0.195 
      epoch: 16 	Loss: 0.192 
      epoch: 17 	Loss: 0.189 
      epoch: 18 	Loss: 0.186 
      epoch: 19 	Loss: 0.183 
      epoch: 20 	Loss: 0.181 

---

    Code
      set.seed(1)
      fit <- brulee_logistic_reg(y ~ ., df_imbal, epochs = 2, verbose = TRUE,
      class_weights = c(a = 12, b = 1), penalty = 0)
    Message <rlang_message>
      epoch: 1 	Loss: 0.113 
      epoch: 2 	Loss: 0.113  x 

