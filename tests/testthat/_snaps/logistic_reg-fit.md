# logistic regression

    Code
      set.seed(1)
      fit <- brulee_logistic_reg(y ~ ., df, epochs = 2, verbose = TRUE, penalty = 0)
    Message <rlang_message>
      epoch: 1 	Loss: 0.17322 
      epoch: 2 	Loss: 0.17322  x 

---

    Code
      fit
    Output
      Logistic regression
      
      1,000 samples, 2 features, 2 classes 
      class weights a=1, b=1 
      batch size: 900 
      validation loss after 1 epoch: 0.17322 

# class weights - logistic regression

    Code
      set.seed(1)
      fit_imbal <- brulee_logistic_reg(y ~ ., df_imbal, verbose = TRUE,
      class_weights = 20, optimizer = "SGD", penalty = 0)
    Message <rlang_message>
      epoch:  1 	Loss: 0.32851 
      epoch:  2 	Loss: 0.30178 
      epoch:  3 	Loss: 0.28238 
      epoch:  4 	Loss: 0.26721 
      epoch:  5 	Loss: 0.25491 
      epoch:  6 	Loss: 0.24467 
      epoch:  7 	Loss: 0.23597 
      epoch:  8 	Loss: 0.22847 
      epoch:  9 	Loss: 0.22192 
      epoch: 10 	Loss: 0.21613 
      epoch: 11 	Loss: 0.21097 
      epoch: 12 	Loss: 0.20633 
      epoch: 13 	Loss: 0.20213 
      epoch: 14 	Loss: 0.19831 
      epoch: 15 	Loss: 0.19482 
      epoch: 16 	Loss: 0.1916 
      epoch: 17 	Loss: 0.18863 
      epoch: 18 	Loss: 0.18587 
      epoch: 19 	Loss: 0.18331 
      epoch: 20 	Loss: 0.18091 

---

    Code
      set.seed(1)
      fit <- brulee_logistic_reg(y ~ ., df_imbal, epochs = 2, verbose = TRUE,
      class_weights = c(a = 12, b = 1), penalty = 0)
    Message <rlang_message>
      epoch: 1 	Loss: 0.11313 
      epoch: 2 	Loss: 0.11313  x 

