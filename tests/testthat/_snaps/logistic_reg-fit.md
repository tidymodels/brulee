# logistic regression

    Code
      set.seed(1)
      logistic_reg_fit_lbfgs <- brulee_logistic_reg(y ~ ., df, epochs = 2, penalty = 0)

---

    Code
      print(logistic_reg_fit_lbfgs)
    Output
      Logistic regression
      
      1,000 samples, 2 features, 2 classes 
      class weights a=1, b=1 
      batch size: 900 
      validation loss after 1 epoch: 0.173 

