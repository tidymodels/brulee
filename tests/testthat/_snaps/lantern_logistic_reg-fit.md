# logistic regression

    Code
      set.seed(1)
      fit <- lantern_logistic_reg(y ~ ., df, epochs = 2, verbose = TRUE)
    Message <message>
      epoch: 1 cross_entropy 0.077 	 
      epoch: 2 cross_entropy 0.077 	 x

---

    Code
      fit
    Output
      Logistic regression
      
      100 samples, 2 features, 2 classes 
      class weights a=1, b=1 
      weight decay: 0.001 
      batch size: 90 
      validation cross_entropy after 1 epochs: 0.077 

