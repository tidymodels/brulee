# different mlp fit interfaces

    Code
      print(mlp_bin_mat_lbfgs_fit)
    Output
      Multilayer perceptron
      
      relu activation
      3 hidden units,  17 model parameters
      1,000 samples, 2 features, 2 classes 
      class weights bad=1, good=1 
      weight decay: 0.001 
      dropout proportion: 0 
      batch size: 900 
      learn rate: 0.01 
      validation loss after 10 epochs: 0.543 

# predictions

    Code
      print(mlp_bin_sgd_fit)
    Output
      Multilayer perceptron
      
      relu activation
      3 hidden units,  17 model parameters
      1,000 samples, 2 features, 2 classes 
      class weights bad=1, good=1 
      weight decay: 0.001 
      dropout proportion: 0 
      batch size: 32 
      learn rate: 0.01 
      validation loss after 10 epochs: 0.299 

# class weights - mlp

    Code
      print(mlp_bin_sgd_fit_20)
    Output
      Multilayer perceptron
      
      relu activation
      3 hidden units,  17 model parameters
      200 samples, 2 features, 2 classes 
      class weights a=20, b= 1 
      weight decay: 0.001 
      dropout proportion: 0 
      batch size: 180 
      learn rate: 0.01 
      validation loss after 100 epochs: 0.438 

---

    Code
      print(mlp_bin_sgd_fit_12)
    Output
      Multilayer perceptron
      
      relu activation
      3 hidden units,  17 model parameters
      200 samples, 2 features, 2 classes 
      class weights a=12, b= 1 
      weight decay: 0.001 
      dropout proportion: 0 
      batch size: 180 
      learn rate: 0.01 
      validation loss after 2 epochs: 0.664 

