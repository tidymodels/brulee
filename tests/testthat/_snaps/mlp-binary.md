# different fit interfaces

    Code
      fit_mat
    Output
      Multilayer perceptron
      
      relu activation
      3 hidden units,  17 model parameters
      1,000 samples, 2 features, 2 classes 
      class weights bad=1, good=1 
      penalty: 0.001 (%lasso: 0)
      dropout proportion: 0 
      batch size: 900 
      optimized via LBFGS 
      validation loss after 10 epochs: 0.29149 

# mlp binary learns something

    Code
      print(model)
    Output
      Multilayer perceptron
      
      relu activation
      5 hidden units,  22 model parameters
      1,000 samples, 1 features, 2 classes 
      class weights FALSE=1, TRUE=1 
      penalty: 0.001 (%lasso: 0)
      dropout proportion: 0 
      batch size: 50 
      optimized via SGD 
      validation loss after 100 epochs: 0.039333 

# class weights - mlp

    Code
      set.seed(1)
      fit_imbal <- brulee_mlp(y ~ ., df_imbal, verbose = TRUE, class_weights = 20,
      learn_rate = 0.1)
    Message <rlang_message>
      epoch:   1 	Loss: 0.081083 
      epoch:   2 	Loss: 0.032033 
      epoch:   3 	Loss: 0.031976 
      epoch:   4 	Loss: 0.033298  x 
      epoch:   5 	Loss: 0.035204  x 
      epoch:   6 	Loss: 0.033884  x 
      epoch:   7 	Loss: 0.033014  x 
      epoch:   8 	Loss: 0.032938  x 

---

    Code
      set.seed(1)
      fit <- brulee_mlp(y ~ ., df_imbal, epochs = 2, verbose = TRUE, class_weights = c(
        a = 12, b = 1))
    Message <rlang_message>
      epoch: 1 	Loss: 0.37215 
      epoch: 2 	Loss: 0.22622 

