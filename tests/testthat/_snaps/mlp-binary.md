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
      learn rate: 0.01 
      validation loss after 10 epochs: 0.294 

# class weights - mlp

    Code
      set.seed(1)
      fit_imbal <- brulee_mlp(y ~ ., df_imbal, verbose = TRUE, class_weights = 20)
    Message
      epoch:   1 learn rate 0.01 	Loss: 0.351 
      epoch:   2 learn rate 0.01 	Loss: 0.224 
      epoch:   3 learn rate 0.01 	Loss: 0.178 
      epoch:   4 learn rate 0.01 	Loss: 0.143 
      epoch:   5 learn rate 0.01 	Loss: 0.115 
      epoch:   6 learn rate 0.01 	Loss: 0.0973 
      epoch:   7 learn rate 0.01 	Loss: 0.0828 
      epoch:   8 learn rate 0.01 	Loss: 0.0717 
      epoch:   9 learn rate 0.01 	Loss: 0.063 
      epoch:  10 learn rate 0.01 	Loss: 0.0558 
      epoch:  11 learn rate 0.01 	Loss: 0.0506 
      epoch:  12 learn rate 0.01 	Loss: 0.0459 
      epoch:  13 learn rate 0.01 	Loss: 0.0422 
      epoch:  14 learn rate 0.01 	Loss: 0.0397 
      epoch:  15 learn rate 0.01 	Loss: 0.037 
      epoch:  16 learn rate 0.01 	Loss: 0.0352 
      epoch:  17 learn rate 0.01 	Loss: 0.0345 
      epoch:  18 learn rate 0.01 	Loss: 0.0334 
      epoch:  19 learn rate 0.01 	Loss: 0.033 
      epoch:  20 learn rate 0.01 	Loss: 0.0324 
      epoch:  21 learn rate 0.01 	Loss: 0.0324 
      epoch:  22 learn rate 0.01 	Loss: 0.0323 
      epoch:  23 learn rate 0.01 	Loss: 0.0323 
      epoch:  24 learn rate 0.01 	Loss: 0.0323 
      epoch:  25 learn rate 0.01 	Loss: 0.0323 
      epoch:  26 learn rate 0.01 	Loss: 0.0323 
      epoch:  27 learn rate 0.01 	Loss: 0.0323 
      epoch:  28 learn rate 0.01 	Loss: 0.0323 
      epoch:  29 learn rate 0.01 	Loss: 0.0323 
      epoch:  30 learn rate 0.01 	Loss: 0.0323 
      epoch:  31 learn rate 0.01 	Loss: 0.0322 
      epoch:  32 learn rate 0.01 	Loss: 0.0322 
      epoch:  33 learn rate 0.01 	Loss: 0.0322 
      epoch:  34 learn rate 0.01 	Loss: 0.0322 
      epoch:  35 learn rate 0.01 	Loss: 0.0322 
      epoch:  36 learn rate 0.01 	Loss: 0.0322 
      epoch:  37 learn rate 0.01 	Loss: 0.0322 
      epoch:  38 learn rate 0.01 	Loss: 0.0322 
      epoch:  39 learn rate 0.01 	Loss: 0.0322 
      epoch:  40 learn rate 0.01 	Loss: 0.0322 
      epoch:  41 learn rate 0.01 	Loss: 0.0322 
      epoch:  42 learn rate 0.01 	Loss: 0.0322 
      epoch:  43 learn rate 0.01 	Loss: 0.0322 
      epoch:  44 learn rate 0.01 	Loss: 0.0322 
      epoch:  45 learn rate 0.01 	Loss: 0.0322 
      epoch:  46 learn rate 0.01 	Loss: 0.0322 
      epoch:  47 learn rate 0.01 	Loss: 0.0322 
      epoch:  48 learn rate 0.01 	Loss: 0.0317 
      epoch:  49 learn rate 0.01 	Loss: 0.0313 
      epoch:  50 learn rate 0.01 	Loss: 0.0314  x 
      epoch:  51 learn rate 0.01 	Loss: 0.031 
      epoch:  52 learn rate 0.01 	Loss: 0.0307 
      epoch:  53 learn rate 0.01 	Loss: 0.0307  x 
      epoch:  54 learn rate 0.01 	Loss: 0.0307  x 
      epoch:  55 learn rate 0.01 	Loss: 0.0307  x 
      epoch:  56 learn rate 0.01 	Loss: 0.0307  x 
      epoch:  57 learn rate 0.01 	Loss: 0.0308  x 

---

    Code
      set.seed(1)
      fit <- brulee_mlp(y ~ ., df_imbal, epochs = 2, verbose = TRUE, class_weights = c(
        a = 12, b = 1))
    Message
      epoch: 1 learn rate 0.01 	Loss: 0.372 
      epoch: 2 learn rate 0.01 	Loss: 0.26 

