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
      epoch:  1 	Loss: 0.92193 
      epoch:  2 	Loss: 0.59453 
      epoch:  3 	Loss: 0.50279 
      epoch:  4 	Loss: 0.46659 
      epoch:  5 	Loss: 0.44918 
      epoch:  6 	Loss: 0.43992 
      epoch:  7 	Loss: 0.4347 
      epoch:  8 	Loss: 0.43165 
      epoch:  9 	Loss: 0.42984 
      epoch: 10 	Loss: 0.42875 
      epoch: 11 	Loss: 0.4281 
      epoch: 12 	Loss: 0.42772 
      epoch: 13 	Loss: 0.4275 
      epoch: 14 	Loss: 0.42738 
      epoch: 15 	Loss: 0.42733 
      epoch: 16 	Loss: 0.42731 
      epoch: 17 	Loss: 0.42731  x 
      epoch: 18 	Loss: 0.42732  x 
      epoch: 19 	Loss: 0.42734  x 
      epoch: 20 	Loss: 0.42736  x 

---

    Code
      set.seed(1)
      fit <- brulee_multinomial_reg(y ~ ., df, epochs = 2, verbose = TRUE,
      class_weights = c(a = 12, b = 1, c = 1), penalty = 0)
    Message <rlang_message>
      epoch: 1 	Loss: 0.38443 
      epoch: 2 	Loss: 0.38443 

