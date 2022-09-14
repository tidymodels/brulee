# multinomial mlp

    Code
      set.seed(1)
      fit <- brulee_mlp(y ~ ., df, epochs = 2, verbose = TRUE)
    Message <rlang_message>
      epoch: 1 learn rate 0.01 	Loss: 1.14 
      epoch: 2 learn rate 0.01 	Loss: 1.14 

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
      validation loss after 2 epochs: 1.14 

# class weights - mlp

    Code
      set.seed(1)
      fit_imbal <- brulee_mlp(y ~ ., df, verbose = TRUE, class_weights = 20)
    Message <rlang_message>
      epoch:   1 learn rate 0.01 	Loss: 1.36 
      epoch:   2 learn rate 0.01 	Loss: 1.35 
      epoch:   3 learn rate 0.01 	Loss: 1.35 
      epoch:   4 learn rate 0.01 	Loss: 1.34 
      epoch:   5 learn rate 0.01 	Loss: 1.33 
      epoch:   6 learn rate 0.01 	Loss: 1.32 
      epoch:   7 learn rate 0.01 	Loss: 1.31 
      epoch:   8 learn rate 0.01 	Loss: 1.3 
      epoch:   9 learn rate 0.01 	Loss: 1.29 
      epoch:  10 learn rate 0.01 	Loss: 1.29 
      epoch:  11 learn rate 0.01 	Loss: 1.28 
      epoch:  12 learn rate 0.01 	Loss: 1.27 
      epoch:  13 learn rate 0.01 	Loss: 1.26 
      epoch:  14 learn rate 0.01 	Loss: 1.25 
      epoch:  15 learn rate 0.01 	Loss: 1.25 
      epoch:  16 learn rate 0.01 	Loss: 1.24 
      epoch:  17 learn rate 0.01 	Loss: 1.23 
      epoch:  18 learn rate 0.01 	Loss: 1.22 
      epoch:  19 learn rate 0.01 	Loss: 1.22 
      epoch:  20 learn rate 0.01 	Loss: 1.21 
      epoch:  21 learn rate 0.01 	Loss: 1.2 
      epoch:  22 learn rate 0.01 	Loss: 1.2 
      epoch:  23 learn rate 0.01 	Loss: 1.19 
      epoch:  24 learn rate 0.01 	Loss: 1.18 
      epoch:  25 learn rate 0.01 	Loss: 1.17 
      epoch:  26 learn rate 0.01 	Loss: 1.17 
      epoch:  27 learn rate 0.01 	Loss: 1.16 
      epoch:  28 learn rate 0.01 	Loss: 1.15 
      epoch:  29 learn rate 0.01 	Loss: 1.15 
      epoch:  30 learn rate 0.01 	Loss: 1.14 
      epoch:  31 learn rate 0.01 	Loss: 1.13 
      epoch:  32 learn rate 0.01 	Loss: 1.13 
      epoch:  33 learn rate 0.01 	Loss: 1.12 
      epoch:  34 learn rate 0.01 	Loss: 1.12 
      epoch:  35 learn rate 0.01 	Loss: 1.11 
      epoch:  36 learn rate 0.01 	Loss: 1.1 
      epoch:  37 learn rate 0.01 	Loss: 1.1 
      epoch:  38 learn rate 0.01 	Loss: 1.09 
      epoch:  39 learn rate 0.01 	Loss: 1.09 
      epoch:  40 learn rate 0.01 	Loss: 1.08 
      epoch:  41 learn rate 0.01 	Loss: 1.07 
      epoch:  42 learn rate 0.01 	Loss: 1.07 
      epoch:  43 learn rate 0.01 	Loss: 1.06 
      epoch:  44 learn rate 0.01 	Loss: 1.06 
      epoch:  45 learn rate 0.01 	Loss: 1.05 
      epoch:  46 learn rate 0.01 	Loss: 1.05 
      epoch:  47 learn rate 0.01 	Loss: 1.04 
      epoch:  48 learn rate 0.01 	Loss: 1.03 
      epoch:  49 learn rate 0.01 	Loss: 1.03 
      epoch:  50 learn rate 0.01 	Loss: 1.02 
      epoch:  51 learn rate 0.01 	Loss: 1.02 
      epoch:  52 learn rate 0.01 	Loss: 1.01 
      epoch:  53 learn rate 0.01 	Loss: 1.01 
      epoch:  54 learn rate 0.01 	Loss: 1 
      epoch:  55 learn rate 0.01 	Loss: 0.998 
      epoch:  56 learn rate 0.01 	Loss: 0.993 
      epoch:  57 learn rate 0.01 	Loss: 0.988 
      epoch:  58 learn rate 0.01 	Loss: 0.983 
      epoch:  59 learn rate 0.01 	Loss: 0.978 
      epoch:  60 learn rate 0.01 	Loss: 0.974 
      epoch:  61 learn rate 0.01 	Loss: 0.969 
      epoch:  62 learn rate 0.01 	Loss: 0.964 
      epoch:  63 learn rate 0.01 	Loss: 0.959 
      epoch:  64 learn rate 0.01 	Loss: 0.955 
      epoch:  65 learn rate 0.01 	Loss: 0.95 
      epoch:  66 learn rate 0.01 	Loss: 0.946 
      epoch:  67 learn rate 0.01 	Loss: 0.941 
      epoch:  68 learn rate 0.01 	Loss: 0.937 
      epoch:  69 learn rate 0.01 	Loss: 0.932 
      epoch:  70 learn rate 0.01 	Loss: 0.928 
      epoch:  71 learn rate 0.01 	Loss: 0.924 
      epoch:  72 learn rate 0.01 	Loss: 0.919 
      epoch:  73 learn rate 0.01 	Loss: 0.915 
      epoch:  74 learn rate 0.01 	Loss: 0.911 
      epoch:  75 learn rate 0.01 	Loss: 0.907 
      epoch:  76 learn rate 0.01 	Loss: 0.903 
      epoch:  77 learn rate 0.01 	Loss: 0.899 
      epoch:  78 learn rate 0.01 	Loss: 0.895 
      epoch:  79 learn rate 0.01 	Loss: 0.891 
      epoch:  80 learn rate 0.01 	Loss: 0.887 
      epoch:  81 learn rate 0.01 	Loss: 0.883 
      epoch:  82 learn rate 0.01 	Loss: 0.879 
      epoch:  83 learn rate 0.01 	Loss: 0.875 
      epoch:  84 learn rate 0.01 	Loss: 0.871 
      epoch:  85 learn rate 0.01 	Loss: 0.867 
      epoch:  86 learn rate 0.01 	Loss: 0.864 
      epoch:  87 learn rate 0.01 	Loss: 0.86 
      epoch:  88 learn rate 0.01 	Loss: 0.856 
      epoch:  89 learn rate 0.01 	Loss: 0.853 
      epoch:  90 learn rate 0.01 	Loss: 0.849 
      epoch:  91 learn rate 0.01 	Loss: 0.846 
      epoch:  92 learn rate 0.01 	Loss: 0.842 
      epoch:  93 learn rate 0.01 	Loss: 0.839 
      epoch:  94 learn rate 0.01 	Loss: 0.835 
      epoch:  95 learn rate 0.01 	Loss: 0.832 
      epoch:  96 learn rate 0.01 	Loss: 0.828 
      epoch:  97 learn rate 0.01 	Loss: 0.825 
      epoch:  98 learn rate 0.01 	Loss: 0.822 
      epoch:  99 learn rate 0.01 	Loss: 0.818 
      epoch: 100 learn rate 0.01 	Loss: 0.815 

---

    Code
      set.seed(1)
      fit <- brulee_mlp(y ~ ., df, epochs = 2, verbose = TRUE, class_weights = c(a = 12,
        b = 1, c = 1))
    Message <rlang_message>
      epoch: 1 learn rate 0.01 	Loss: 0.728 
      epoch: 2 learn rate 0.01 	Loss: 0.725 

