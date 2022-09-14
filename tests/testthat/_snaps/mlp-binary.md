# different fit interfaces

    Code
      fit_mat
    Output
      Multilayer perceptron
      
      relu activation
      3 hidden units,  17 model parameters
      1,000 samples, 2 features, 2 classes 
      class weights bad=1, good=1 
      weight decay: 0.001 
      dropout proportion: 0 
      batch size: 900 
      validation loss after 10 epochs: 0.29149 

# class weights - mlp

    Code
      set.seed(1)
      fit_imbal <- brulee_mlp(y ~ ., df_imbal, verbose = TRUE, class_weights = 20)
    Message <rlang_message>
      epoch:   1 	Loss: 0.35071 
      epoch:   2 	Loss: 0.2112 
      epoch:   3 	Loss: 0.15631 
      epoch:   4 	Loss: 0.11551 
      epoch:   5 	Loss: 0.095834 
      epoch:   6 	Loss: 0.082102 
      epoch:   7 	Loss: 0.071657 
      epoch:   8 	Loss: 0.063983 
      epoch:   9 	Loss: 0.056966 
      epoch:  10 	Loss: 0.051545 
      epoch:  11 	Loss: 0.047123 
      epoch:  12 	Loss: 0.044421 
      epoch:  13 	Loss: 0.04217 
      epoch:  14 	Loss: 0.039573 
      epoch:  15 	Loss: 0.037483 
      epoch:  16 	Loss: 0.035667 
      epoch:  17 	Loss: 0.034444 
      epoch:  18 	Loss: 0.03298 
      epoch:  19 	Loss: 0.032816 
      epoch:  20 	Loss: 0.031928 
      epoch:  21 	Loss: 0.03154 
      epoch:  22 	Loss: 0.030719 
      epoch:  23 	Loss: 0.030044 
      epoch:  24 	Loss: 0.02983 
      epoch:  25 	Loss: 0.029393 
      epoch:  26 	Loss: 0.029271 
      epoch:  27 	Loss: 0.029249 
      epoch:  28 	Loss: 0.029297  x 
      epoch:  29 	Loss: 0.029283  x 
      epoch:  30 	Loss: 0.02927  x 
      epoch:  31 	Loss: 0.029285  x 
      epoch:  32 	Loss: 0.029233 
      epoch:  33 	Loss: 0.029234  x 
      epoch:  34 	Loss: 0.029162 
      epoch:  35 	Loss: 0.029125 
      epoch:  36 	Loss: 0.029124 
      epoch:  37 	Loss: 0.029124 
      epoch:  38 	Loss: 0.029124  x 
      epoch:  39 	Loss: 0.029124  x 
      epoch:  40 	Loss: 0.029124  x 
      epoch:  41 	Loss: 0.029124  x 
      epoch:  42 	Loss: 0.029125  x 

---

    Code
      set.seed(1)
      fit <- brulee_mlp(y ~ ., df_imbal, epochs = 2, verbose = TRUE, class_weights = c(
        a = 12, b = 1))
    Message <rlang_message>
      epoch: 1 	Loss: 0.37215 
      epoch: 2 	Loss: 0.22622 

