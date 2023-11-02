# multinomial mlp

    Code
      print(mlp_mlt_mat_lbfgs_fit)
    Output
      Multilayer perceptron
      
      relu activation
      3 hidden units,  21 model parameters
      10,000 samples, 2 features, 3 classes 
      class weights a=1, b=1, c=1 
      weight decay: 0.001 
      dropout proportion: 0 
      batch size: 9000 
      learn rate: 0.01 
      validation loss after 2 epochs: 1.05 

# class weights - mlp

    Code
      print(mlp_bin_lbfgs_fit_20)
    Output
      Multilayer perceptron
      
      relu activation
      3 hidden units,  21 model parameters
      10,000 samples, 2 features, 3 classes 
      class weights a= 1, b= 1, c=20 
      weight decay: 0.001 
      dropout proportion: 0 
      batch size: 9000 
      learn rate: 0.01 
      validation loss after 22 epochs: 0.426 

---

    Code
      print(mlp_bin_lbfgs_fit_12)
    Output
      Multilayer perceptron
      
      relu activation
      3 hidden units,  21 model parameters
      10,000 samples, 2 features, 3 classes 
      class weights a=12, b= 1, c= 1 
      weight decay: 0.001 
      dropout proportion: 0 
      batch size: 9000 
      learn rate: 0.01 
      validation loss after 2 epochs: 0.435 

