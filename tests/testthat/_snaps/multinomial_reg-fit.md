# multinomial regression

    Code
      set.seed(1)
      fit <- brulee_multinomial_reg(y ~ ., df, epochs = 2, verbose = TRUE, penalty = 0)
    Message <rlang_message>
      epoch: 1 	Loss: 1.02 
      epoch: 2 	Loss: 1.02 

---

    Code
      fit
    Output
      Multinomial regression
      
      10,000 samples, 2 features, 3 classes 
      class weights a=1, b=1, c=1 
      batch size: 9000 
      validation loss after 2 epochs: 1.02 

# class weights - multinomial regression

    Code
      set.seed(1)
      fit_imbal <- brulee_multinomial_reg(y ~ ., df, verbose = TRUE, class_weights = 20,
      optimizer = "SGD")
    Message <rlang_message>
      epoch:  1 	Loss: 0.921 
      epoch:  2 	Loss: 0.593 
      epoch:  3 	Loss: 0.501 
      epoch:  4 	Loss: 0.465 
      epoch:  5 	Loss: 0.447 
      epoch:  6 	Loss: 0.438 
      epoch:  7 	Loss: 0.433 
      epoch:  8 	Loss: 0.43 
      epoch:  9 	Loss: 0.428 
      epoch: 10 	Loss: 0.427 
      epoch: 11 	Loss: 0.426 
      epoch: 12 	Loss: 0.426 
      epoch: 13 	Loss: 0.426 
      epoch: 14 	Loss: 0.426 
      epoch: 15 	Loss: 0.425 
      epoch: 16 	Loss: 0.425 
      epoch: 17 	Loss: 0.425  x 
      epoch: 18 	Loss: 0.425  x 
      epoch: 19 	Loss: 0.426  x 
      epoch: 20 	Loss: 0.426  x 

---

    Code
      set.seed(1)
      fit <- brulee_multinomial_reg(y ~ ., df, epochs = 2, verbose = TRUE,
      class_weights = c(a = 12, b = 1, c = 1), penalty = 0)
    Message <rlang_message>
      epoch: 1 	Loss: 0.384 
      epoch: 2 	Loss: 0.384 

