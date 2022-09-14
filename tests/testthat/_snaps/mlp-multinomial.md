# multinomial mlp

    Code
      set.seed(1)
      fit <- brulee_mlp(y ~ ., df, epochs = 2, verbose = TRUE)
    Message <rlang_message>
      epoch: 1 	Loss: 1.0845 
      epoch: 2 	Loss: 1.0425 

---

    Code
      fit
    Output
      Multilayer perceptron
      
      relu activation
      3 hidden units,  21 model parameters
      10,000 samples, 2 features, 3 classes 
      class weights a=1, b=1, c=1 
      weight decay: 0.001 
      dropout proportion: 0 
      batch size: 9000 
      validation loss after 2 epochs: 1.0425 

