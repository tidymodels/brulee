# multinomial regression

    Code
      set.seed(1)
      fit <- brulee_multinomial_reg(y ~ ., df, epochs = 2, verbose = TRUE, penalty = 0)
    Message <rlang_message>
      epoch: 1 	Loss: 1.0243 
      epoch: 2 	Loss: 1.0243 

---

    Code
      fit
    Output
      Multinomial regression
      
      10,000 samples, 2 features, 3 classes 
      class weights a=1, b=1, c=1 
      batch size: 9000 
      validation loss after 2 epochs: 1.0243 

# class weights - multinomial regression

    Code
      set.seed(1)
      fit_imbal <- brulee_multinomial_reg(y ~ ., df, verbose = TRUE, class_weights = 20,
      optimizer = "SGD")
    Message <rlang_message>
      epoch:  1 	Loss: 0.92056 
      epoch:  2 	Loss: 0.59318 
      epoch:  3 	Loss: 0.50124 
      epoch:  4 	Loss: 0.46489 
      epoch:  5 	Loss: 0.44743 
      epoch:  6 	Loss: 0.43814 
      epoch:  7 	Loss: 0.43291 
      epoch:  8 	Loss: 0.42985 
      epoch:  9 	Loss: 0.42803 
      epoch: 10 	Loss: 0.42693 
      epoch: 11 	Loss: 0.42628 
      epoch: 12 	Loss: 0.42589 
      epoch: 13 	Loss: 0.42567 
      epoch: 14 	Loss: 0.42555 
      epoch: 15 	Loss: 0.42549 
      epoch: 16 	Loss: 0.42547 
      epoch: 17 	Loss: 0.42547  x 
      epoch: 18 	Loss: 0.42549  x 
      epoch: 19 	Loss: 0.4255  x 
      epoch: 20 	Loss: 0.42552  x 

---

    Code
      set.seed(1)
      fit <- brulee_multinomial_reg(y ~ ., df, epochs = 2, verbose = TRUE,
      class_weights = c(a = 12, b = 1, c = 1), penalty = 0)
    Message <rlang_message>
      epoch: 1 	Loss: 0.38443 
      epoch: 2 	Loss: 0.38443 

