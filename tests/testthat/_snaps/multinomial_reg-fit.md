# multinomial regression

    Code
      set.seed(1)
      fit <- brulee_multinomial_reg(y ~ ., df, epochs = 2, verbose = TRUE)
    Message <message>
      epoch: 1 	Loss: 1.0243 
      epoch: 2 	Loss: 1.0243 

---

    Code
      fit
    Output
      Multinomial regression
      
      10,000 samples, 2 features, 3 classes 
      class weights a=1, b=1, c=1 
      weight decay: 0.001 
      batch size: 9000 
      validation loss after 2 epochs: 1.0243 

# class weights - multinomial regression

    Code
      set.seed(1)
      fit_imbal <- brulee_multinomial_reg(y ~ ., df, verbose = TRUE, class_weights = 20,
      optimizer = "SGD")
    Message <message>
      epoch:  1 	Loss: 0.92035 
      epoch:  2 	Loss: 0.5928 
      epoch:  3 	Loss: 0.50078 
      epoch:  4 	Loss: 0.4644 
      epoch:  5 	Loss: 0.44692 
      epoch:  6 	Loss: 0.43763 
      epoch:  7 	Loss: 0.43238 
      epoch:  8 	Loss: 0.42932 
      epoch:  9 	Loss: 0.42749 
      epoch: 10 	Loss: 0.42639 
      epoch: 11 	Loss: 0.42573 
      epoch: 12 	Loss: 0.42534 
      epoch: 13 	Loss: 0.42511 
      epoch: 14 	Loss: 0.42499 
      epoch: 15 	Loss: 0.42493 
      epoch: 16 	Loss: 0.42491 
      epoch: 17 	Loss: 0.42491 
      epoch: 18 	Loss: 0.42492  x 
      epoch: 19 	Loss: 0.42494  x 
      epoch: 20 	Loss: 0.42495  x 

---

    Code
      set.seed(1)
      fit <- brulee_multinomial_reg(y ~ ., df, epochs = 2, verbose = TRUE,
      class_weights = c(a = 12, b = 1, c = 1))
    Message <message>
      epoch: 1 	Loss: 0.38443 
      epoch: 2 	Loss: 0.38443 

