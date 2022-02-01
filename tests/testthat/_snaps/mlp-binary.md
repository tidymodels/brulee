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
      validation loss after 10 epochs: 0.54321 

# class weights - mlp

    Code
      set.seed(1)
      fit_imbal <- brulee_mlp(y ~ ., df_imbal, verbose = TRUE, class_weights = 20)
    Message <rlang_message>
      epoch:   1 	Loss: 0.57753 
      epoch:   2 	Loss: 0.57439 
      epoch:   3 	Loss: 0.57134 
      epoch:   4 	Loss: 0.56836 
      epoch:   5 	Loss: 0.56547 
      epoch:   6 	Loss: 0.56264 
      epoch:   7 	Loss: 0.55981 
      epoch:   8 	Loss: 0.55706 
      epoch:   9 	Loss: 0.55437 
      epoch:  10 	Loss: 0.55176 
      epoch:  11 	Loss: 0.54921 
      epoch:  12 	Loss: 0.54672 
      epoch:  13 	Loss: 0.54429 
      epoch:  14 	Loss: 0.54192 
      epoch:  15 	Loss: 0.5396 
      epoch:  16 	Loss: 0.53733 
      epoch:  17 	Loss: 0.53511 
      epoch:  18 	Loss: 0.53293 
      epoch:  19 	Loss: 0.5308 
      epoch:  20 	Loss: 0.52872 
      epoch:  21 	Loss: 0.52663 
      epoch:  22 	Loss: 0.52459 
      epoch:  23 	Loss: 0.52259 
      epoch:  24 	Loss: 0.52063 
      epoch:  25 	Loss: 0.51871 
      epoch:  26 	Loss: 0.51684 
      epoch:  27 	Loss: 0.515 
      epoch:  28 	Loss: 0.51321 
      epoch:  29 	Loss: 0.51144 
      epoch:  30 	Loss: 0.50972 
      epoch:  31 	Loss: 0.50803 
      epoch:  32 	Loss: 0.50637 
      epoch:  33 	Loss: 0.50475 
      epoch:  34 	Loss: 0.50317 
      epoch:  35 	Loss: 0.50163 
      epoch:  36 	Loss: 0.50011 
      epoch:  37 	Loss: 0.49863 
      epoch:  38 	Loss: 0.49717 
      epoch:  39 	Loss: 0.49568 
      epoch:  40 	Loss: 0.49424 
      epoch:  41 	Loss: 0.49283 
      epoch:  42 	Loss: 0.49145 
      epoch:  43 	Loss: 0.49009 
      epoch:  44 	Loss: 0.48876 
      epoch:  45 	Loss: 0.48745 
      epoch:  46 	Loss: 0.48616 
      epoch:  47 	Loss: 0.48489 
      epoch:  48 	Loss: 0.48365 
      epoch:  49 	Loss: 0.48242 
      epoch:  50 	Loss: 0.48121 
      epoch:  51 	Loss: 0.48002 
      epoch:  52 	Loss: 0.47885 
      epoch:  53 	Loss: 0.4777 
      epoch:  54 	Loss: 0.47657 
      epoch:  55 	Loss: 0.47546 
      epoch:  56 	Loss: 0.47436 
      epoch:  57 	Loss: 0.47327 
      epoch:  58 	Loss: 0.47221 
      epoch:  59 	Loss: 0.47115 
      epoch:  60 	Loss: 0.47012 
      epoch:  61 	Loss: 0.46911 
      epoch:  62 	Loss: 0.46811 
      epoch:  63 	Loss: 0.46712 
      epoch:  64 	Loss: 0.46615 
      epoch:  65 	Loss: 0.46519 
      epoch:  66 	Loss: 0.46424 
      epoch:  67 	Loss: 0.46331 
      epoch:  68 	Loss: 0.46239 
      epoch:  69 	Loss: 0.46149 
      epoch:  70 	Loss: 0.46059 
      epoch:  71 	Loss: 0.4597 
      epoch:  72 	Loss: 0.45883 
      epoch:  73 	Loss: 0.45797 
      epoch:  74 	Loss: 0.45712 
      epoch:  75 	Loss: 0.45627 
      epoch:  76 	Loss: 0.45544 
      epoch:  77 	Loss: 0.45462 
      epoch:  78 	Loss: 0.45381 
      epoch:  79 	Loss: 0.453 
      epoch:  80 	Loss: 0.45221 
      epoch:  81 	Loss: 0.45142 
      epoch:  82 	Loss: 0.45065 
      epoch:  83 	Loss: 0.44988 
      epoch:  84 	Loss: 0.44914 
      epoch:  85 	Loss: 0.44841 
      epoch:  86 	Loss: 0.44768 
      epoch:  87 	Loss: 0.44697 
      epoch:  88 	Loss: 0.44626 
      epoch:  89 	Loss: 0.44555 
      epoch:  90 	Loss: 0.44486 
      epoch:  91 	Loss: 0.44417 
      epoch:  92 	Loss: 0.44348 
      epoch:  93 	Loss: 0.44281 
      epoch:  94 	Loss: 0.44215 
      epoch:  95 	Loss: 0.44149 
      epoch:  96 	Loss: 0.44084 
      epoch:  97 	Loss: 0.4402 
      epoch:  98 	Loss: 0.43956 
      epoch:  99 	Loss: 0.43893 
      epoch: 100 	Loss: 0.4383 

---

    Code
      set.seed(1)
      fit <- brulee_mlp(y ~ ., df_imbal, epochs = 2, verbose = TRUE, class_weights = c(
        a = 12, b = 1))
    Message <rlang_message>
      epoch: 1 	Loss: 0.6662 
      epoch: 2 	Loss: 0.66393 

