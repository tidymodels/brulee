# multinomial mlp

    Code
      set.seed(1)
      fit <- brulee_mlp(y ~ ., df, epochs = 2, verbose = TRUE)
    Message <rlang_message>
      epoch: 1 	Loss: 1.1435 
      epoch: 2 	Loss: 1.1429 

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
      validation loss after 2 epochs: 1.1429 

# class weights - mlp

    Code
      set.seed(1)
      fit_imbal <- brulee_mlp(y ~ ., df, verbose = TRUE, class_weights = 20,
      optimizer = "SGD")
    Message <rlang_message>
      epoch:   1 	Loss: 1.3634 
      epoch:   2 	Loss: 1.3545 
      epoch:   3 	Loss: 1.3457 
      epoch:   4 	Loss: 1.3369 
      epoch:   5 	Loss: 1.3283 
      epoch:   6 	Loss: 1.3198 
      epoch:   7 	Loss: 1.3113 
      epoch:   8 	Loss: 1.303 
      epoch:   9 	Loss: 1.2947 
      epoch:  10 	Loss: 1.2866 
      epoch:  11 	Loss: 1.2786 
      epoch:  12 	Loss: 1.2706 
      epoch:  13 	Loss: 1.2627 
      epoch:  14 	Loss: 1.2549 
      epoch:  15 	Loss: 1.2472 
      epoch:  16 	Loss: 1.2396 
      epoch:  17 	Loss: 1.2321 
      epoch:  18 	Loss: 1.2247 
      epoch:  19 	Loss: 1.2173 
      epoch:  20 	Loss: 1.21 
      epoch:  21 	Loss: 1.2028 
      epoch:  22 	Loss: 1.1957 
      epoch:  23 	Loss: 1.1887 
      epoch:  24 	Loss: 1.1817 
      epoch:  25 	Loss: 1.1748 
      epoch:  26 	Loss: 1.168 
      epoch:  27 	Loss: 1.1612 
      epoch:  28 	Loss: 1.1545 
      epoch:  29 	Loss: 1.1479 
      epoch:  30 	Loss: 1.1414 
      epoch:  31 	Loss: 1.1349 
      epoch:  32 	Loss: 1.1285 
      epoch:  33 	Loss: 1.1222 
      epoch:  34 	Loss: 1.1159 
      epoch:  35 	Loss: 1.1097 
      epoch:  36 	Loss: 1.1036 
      epoch:  37 	Loss: 1.0975 
      epoch:  38 	Loss: 1.0915 
      epoch:  39 	Loss: 1.0855 
      epoch:  40 	Loss: 1.0797 
      epoch:  41 	Loss: 1.0738 
      epoch:  42 	Loss: 1.0681 
      epoch:  43 	Loss: 1.0624 
      epoch:  44 	Loss: 1.0567 
      epoch:  45 	Loss: 1.0511 
      epoch:  46 	Loss: 1.0456 
      epoch:  47 	Loss: 1.0401 
      epoch:  48 	Loss: 1.0347 
      epoch:  49 	Loss: 1.0293 
      epoch:  50 	Loss: 1.024 
      epoch:  51 	Loss: 1.0187 
      epoch:  52 	Loss: 1.0135 
      epoch:  53 	Loss: 1.0083 
      epoch:  54 	Loss: 1.0032 
      epoch:  55 	Loss: 0.99815 
      epoch:  56 	Loss: 0.99315 
      epoch:  57 	Loss: 0.98819 
      epoch:  58 	Loss: 0.98328 
      epoch:  59 	Loss: 0.97842 
      epoch:  60 	Loss: 0.97361 
      epoch:  61 	Loss: 0.96884 
      epoch:  62 	Loss: 0.96412 
      epoch:  63 	Loss: 0.95945 
      epoch:  64 	Loss: 0.95483 
      epoch:  65 	Loss: 0.95024 
      epoch:  66 	Loss: 0.94571 
      epoch:  67 	Loss: 0.94122 
      epoch:  68 	Loss: 0.93677 
      epoch:  69 	Loss: 0.93237 
      epoch:  70 	Loss: 0.92801 
      epoch:  71 	Loss: 0.92369 
      epoch:  72 	Loss: 0.91942 
      epoch:  73 	Loss: 0.91518 
      epoch:  74 	Loss: 0.91099 
      epoch:  75 	Loss: 0.90684 
      epoch:  76 	Loss: 0.90273 
      epoch:  77 	Loss: 0.89866 
      epoch:  78 	Loss: 0.89462 
      epoch:  79 	Loss: 0.89063 
      epoch:  80 	Loss: 0.88667 
      epoch:  81 	Loss: 0.88276 
      epoch:  82 	Loss: 0.87888 
      epoch:  83 	Loss: 0.87504 
      epoch:  84 	Loss: 0.87123 
      epoch:  85 	Loss: 0.86746 
      epoch:  86 	Loss: 0.86373 
      epoch:  87 	Loss: 0.86004 
      epoch:  88 	Loss: 0.85637 
      epoch:  89 	Loss: 0.85275 
      epoch:  90 	Loss: 0.84916 
      epoch:  91 	Loss: 0.8456 
      epoch:  92 	Loss: 0.84208 
      epoch:  93 	Loss: 0.83859 
      epoch:  94 	Loss: 0.83514 
      epoch:  95 	Loss: 0.83171 
      epoch:  96 	Loss: 0.82832 
      epoch:  97 	Loss: 0.82497 
      epoch:  98 	Loss: 0.82164 
      epoch:  99 	Loss: 0.81835 
      epoch: 100 	Loss: 0.81509 

---

    Code
      set.seed(1)
      fit <- brulee_mlp(y ~ ., df, epochs = 2, verbose = TRUE, class_weights = c(a = 12,
        b = 1, c = 1))
    Message <rlang_message>
      epoch: 1 	Loss: 0.72788 
      epoch: 2 	Loss: 0.72504 

