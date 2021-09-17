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
      fit <- lantern_logistic_reg(y ~ ., df, epochs = 2, verbose = TRUE,
      class_weights = 12)
    Message <message>
      epoch: 1 	Loss: 0.011317 
      epoch: 2 	Loss: 0.011317  x 

---

    Code
      set.seed(1)
      fit <- lantern_logistic_reg(y ~ ., df, epochs = 2, verbose = TRUE,
      class_weights = c(a = 12, b = 1))
    Message <message>
      epoch: 1 	Loss: 0.011317 
      epoch: 2 	Loss: 0.011317  x 

