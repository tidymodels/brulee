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
      epoch:   5 learn rate 0.01 	Loss: 0.116 
      epoch:   6 learn rate 0.01 	Loss: 0.0971 
      epoch:   7 learn rate 0.01 	Loss: 0.0823 
      epoch:   8 learn rate 0.01 	Loss: 0.0713 
      epoch:   9 learn rate 0.01 	Loss: 0.0631 
      epoch:  10 learn rate 0.01 	Loss: 0.0564 
      epoch:  11 learn rate 0.01 	Loss: 0.0503 
      epoch:  12 learn rate 0.01 	Loss: 0.046 
      epoch:  13 learn rate 0.01 	Loss: 0.0427 
      epoch:  14 learn rate 0.01 	Loss: 0.0401 
      epoch:  15 learn rate 0.01 	Loss: 0.038 
      epoch:  16 learn rate 0.01 	Loss: 0.0358 
      epoch:  17 learn rate 0.01 	Loss: 0.0342 
      epoch:  18 learn rate 0.01 	Loss: 0.0334 
      epoch:  19 learn rate 0.01 	Loss: 0.0334 
      epoch:  20 learn rate 0.01 	Loss: 0.0334 
      epoch:  21 learn rate 0.01 	Loss: 0.0334 
      epoch:  22 learn rate 0.01 	Loss: 0.0334 
      epoch:  23 learn rate 0.01 	Loss: 0.0333 
      epoch:  24 learn rate 0.01 	Loss: 0.0333 
      epoch:  25 learn rate 0.01 	Loss: 0.0333 
      epoch:  26 learn rate 0.01 	Loss: 0.0333 
      epoch:  27 learn rate 0.01 	Loss: 0.0333 
      epoch:  28 learn rate 0.01 	Loss: 0.0333 
      epoch:  29 learn rate 0.01 	Loss: 0.0333 
      epoch:  30 learn rate 0.01 	Loss: 0.0333 
      epoch:  31 learn rate 0.01 	Loss: 0.0332 
      epoch:  32 learn rate 0.01 	Loss: 0.0332 
      epoch:  33 learn rate 0.01 	Loss: 0.0332 
      epoch:  34 learn rate 0.01 	Loss: 0.0332 
      epoch:  35 learn rate 0.01 	Loss: 0.0332 
      epoch:  36 learn rate 0.01 	Loss: 0.0332 
      epoch:  37 learn rate 0.01 	Loss: 0.0332 
      epoch:  38 learn rate 0.01 	Loss: 0.0332 
      epoch:  39 learn rate 0.01 	Loss: 0.0327 
      epoch:  40 learn rate 0.01 	Loss: 0.0327 
      epoch:  41 learn rate 0.01 	Loss: 0.0327 
      epoch:  42 learn rate 0.01 	Loss: 0.0327 
      epoch:  43 learn rate 0.01 	Loss: 0.0327 
      epoch:  44 learn rate 0.01 	Loss: 0.0327 
      epoch:  45 learn rate 0.01 	Loss: 0.0327 
      epoch:  46 learn rate 0.01 	Loss: 0.0326 
      epoch:  47 learn rate 0.01 	Loss: 0.0326 
      epoch:  48 learn rate 0.01 	Loss: 0.0326 
      epoch:  49 learn rate 0.01 	Loss: 0.0326 
      epoch:  50 learn rate 0.01 	Loss: 0.0326 
      epoch:  51 learn rate 0.01 	Loss: 0.0326 
      epoch:  52 learn rate 0.01 	Loss: 0.0326 
      epoch:  53 learn rate 0.01 	Loss: 0.0326 
      epoch:  54 learn rate 0.01 	Loss: 0.0326 
      epoch:  55 learn rate 0.01 	Loss: 0.0326 
      epoch:  56 learn rate 0.01 	Loss: 0.0326 
      epoch:  57 learn rate 0.01 	Loss: 0.0326 
      epoch:  58 learn rate 0.01 	Loss: 0.0326 
      epoch:  59 learn rate 0.01 	Loss: 0.0325 
      epoch:  60 learn rate 0.01 	Loss: 0.0325 
      epoch:  61 learn rate 0.01 	Loss: 0.0325 
      epoch:  62 learn rate 0.01 	Loss: 0.0325 
      epoch:  63 learn rate 0.01 	Loss: 0.0325 
      epoch:  64 learn rate 0.01 	Loss: 0.0325 
      epoch:  65 learn rate 0.01 	Loss: 0.0325 
      epoch:  66 learn rate 0.01 	Loss: 0.0325 
      epoch:  67 learn rate 0.01 	Loss: 0.0325 
      epoch:  68 learn rate 0.01 	Loss: 0.0325 
      epoch:  69 learn rate 0.01 	Loss: 0.0325 
      epoch:  70 learn rate 0.01 	Loss: 0.0325 
      epoch:  71 learn rate 0.01 	Loss: 0.0325 
      epoch:  72 learn rate 0.01 	Loss: 0.0325 
      epoch:  73 learn rate 0.01 	Loss: 0.0325 
      epoch:  74 learn rate 0.01 	Loss: 0.0325 
      epoch:  75 learn rate 0.01 	Loss: 0.0325 
      epoch:  76 learn rate 0.01 	Loss: 0.0323 
      epoch:  77 learn rate 0.01 	Loss: 0.0319 
      epoch:  78 learn rate 0.01 	Loss: 0.0319 
      epoch:  79 learn rate 0.01 	Loss: 0.0319 
      epoch:  80 learn rate 0.01 	Loss: 0.0319 
      epoch:  81 learn rate 0.01 	Loss: 0.0319 
      epoch:  82 learn rate 0.01 	Loss: 0.0319 
      epoch:  83 learn rate 0.01 	Loss: 0.0318 
      epoch:  84 learn rate 0.01 	Loss: 0.0318 
      epoch:  85 learn rate 0.01 	Loss: 0.0318 
      epoch:  86 learn rate 0.01 	Loss: 0.0318 
      epoch:  87 learn rate 0.01 	Loss: 0.0318 
      epoch:  88 learn rate 0.01 	Loss: 0.0318 
      epoch:  89 learn rate 0.01 	Loss: 0.0318 
      epoch:  90 learn rate 0.01 	Loss: 0.0318 
      epoch:  91 learn rate 0.01 	Loss: 0.0318 
      epoch:  92 learn rate 0.01 	Loss: 0.0318 
      epoch:  93 learn rate 0.01 	Loss: 0.0318 
      epoch:  94 learn rate 0.01 	Loss: 0.0318 
      epoch:  95 learn rate 0.01 	Loss: 0.0318 
      epoch:  96 learn rate 0.01 	Loss: 0.0318 
      epoch:  97 learn rate 0.01 	Loss: 0.0318 
      epoch:  98 learn rate 0.01 	Loss: 0.0318 
      epoch:  99 learn rate 0.01 	Loss: 0.0318 
      epoch: 100 learn rate 0.01 	Loss: 0.0318 

---

    Code
      set.seed(1)
      fit <- brulee_mlp(y ~ ., df_imbal, epochs = 2, verbose = TRUE, class_weights = c(
        a = 12, b = 1))
    Message
      epoch: 1 learn rate 0.01 	Loss: 0.372 
      epoch: 2 learn rate 0.01 	Loss: 0.26 

