# multinomial mlp

    Code
      set.seed(1)
      fit <- brulee_mlp(y ~ ., df, epochs = 2, verbose = TRUE)
    Message
      epoch: 1 learn rate 0.01 	Loss: 1.06 
      epoch: 2 learn rate 0.01 	Loss: 1.04 

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
      learn rate: 0.01 
      validation loss after 2 epochs: 1.04 

# class weights - mlp

    Code
      set.seed(1)
      fit_imbal <- brulee_mlp(y ~ ., df, verbose = TRUE, class_weights = 20)
    Message
      epoch:   1 learn rate 0.01 	Loss: 0.675 
      epoch:   2 learn rate 0.01 	Loss: 0.508 
      epoch:   3 learn rate 0.01 	Loss: 0.467 
      epoch:   4 learn rate 0.01 	Loss: 0.453 
      epoch:   5 learn rate 0.01 	Loss: 0.444 
      epoch:   6 learn rate 0.01 	Loss: 0.438 
      epoch:   7 learn rate 0.01 	Loss: 0.434 
      epoch:   8 learn rate 0.01 	Loss: 0.432 
      epoch:   9 learn rate 0.01 	Loss: 0.43 
      epoch:  10 learn rate 0.01 	Loss: 0.429 
      epoch:  11 learn rate 0.01 	Loss: 0.428 
      epoch:  12 learn rate 0.01 	Loss: 0.427 
      epoch:  13 learn rate 0.01 	Loss: 0.427 
      epoch:  14 learn rate 0.01 	Loss: 0.427 
      epoch:  15 learn rate 0.01 	Loss: 0.427 
      epoch:  16 learn rate 0.01 	Loss: 0.427 
      epoch:  17 learn rate 0.01 	Loss: 0.427 
      epoch:  18 learn rate 0.01 	Loss: 0.427 
      epoch:  19 learn rate 0.01 	Loss: 0.427 
      epoch:  20 learn rate 0.01 	Loss: 0.427 
      epoch:  21 learn rate 0.01 	Loss: 0.426 
      epoch:  22 learn rate 0.01 	Loss: 0.426 
      epoch:  23 learn rate 0.01 	Loss: 0.426 
      epoch:  24 learn rate 0.01 	Loss: 0.426 
      epoch:  25 learn rate 0.01 	Loss: 0.426 
      epoch:  26 learn rate 0.01 	Loss: 0.426 
      epoch:  27 learn rate 0.01 	Loss: 0.426 
      epoch:  28 learn rate 0.01 	Loss: 0.426 
      epoch:  29 learn rate 0.01 	Loss: 0.426 
      epoch:  30 learn rate 0.01 	Loss: 0.426 
      epoch:  31 learn rate 0.01 	Loss: 0.426 
      epoch:  32 learn rate 0.01 	Loss: 0.426 
      epoch:  33 learn rate 0.01 	Loss: 0.426 
      epoch:  34 learn rate 0.01 	Loss: 0.426 
      epoch:  35 learn rate 0.01 	Loss: 0.426 
      epoch:  36 learn rate 0.01 	Loss: 0.426 
      epoch:  37 learn rate 0.01 	Loss: 0.426 
      epoch:  38 learn rate 0.01 	Loss: 0.426  x 
      epoch:  39 learn rate 0.01 	Loss: 0.426  x 
      epoch:  40 learn rate 0.01 	Loss: 0.426 
      epoch:  41 learn rate 0.01 	Loss: 0.426 
      epoch:  42 learn rate 0.01 	Loss: 0.426 
      epoch:  43 learn rate 0.01 	Loss: 0.426 
      epoch:  44 learn rate 0.01 	Loss: 0.426 
      epoch:  45 learn rate 0.01 	Loss: 0.426 
      epoch:  46 learn rate 0.01 	Loss: 0.426 
      epoch:  47 learn rate 0.01 	Loss: 0.426 
      epoch:  48 learn rate 0.01 	Loss: 0.426 
      epoch:  49 learn rate 0.01 	Loss: 0.426 
      epoch:  50 learn rate 0.01 	Loss: 0.426  x 
      epoch:  51 learn rate 0.01 	Loss: 0.426  x 
      epoch:  52 learn rate 0.01 	Loss: 0.426 
      epoch:  53 learn rate 0.01 	Loss: 0.426 
      epoch:  54 learn rate 0.01 	Loss: 0.426 
      epoch:  55 learn rate 0.01 	Loss: 0.426  x 
      epoch:  56 learn rate 0.01 	Loss: 0.426 
      epoch:  57 learn rate 0.01 	Loss: 0.426 
      epoch:  58 learn rate 0.01 	Loss: 0.426  x 
      epoch:  59 learn rate 0.01 	Loss: 0.426 
      epoch:  60 learn rate 0.01 	Loss: 0.426 
      epoch:  61 learn rate 0.01 	Loss: 0.426 
      epoch:  62 learn rate 0.01 	Loss: 0.426  x 
      epoch:  63 learn rate 0.01 	Loss: 0.426  x 
      epoch:  64 learn rate 0.01 	Loss: 0.426  x 
      epoch:  65 learn rate 0.01 	Loss: 0.426  x 
      epoch:  66 learn rate 0.01 	Loss: 0.426  x 

---

    Code
      set.seed(1)
      fit <- brulee_mlp(y ~ ., df, epochs = 2, verbose = TRUE, class_weights = c(a = 12,
        b = 1, c = 1))
    Message
      epoch: 1 learn rate 0.01 	Loss: 0.478 
      epoch: 2 learn rate 0.01 	Loss: 0.435 

