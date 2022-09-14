# multinomial mlp

    Code
      set.seed(1)
      fit <- brulee_mlp(y ~ ., df, epochs = 2, verbose = TRUE)
    Message <rlang_message>
      epoch: 1 learn rate 0.01 	Loss: 1.1435 
      epoch: 2 learn rate 0.01 	Loss: 1.1429 

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
      validation loss after 2 epochs: 1.1429 

# class weights - mlp

    Code
      set.seed(1)
      fit_imbal <- brulee_mlp(y ~ ., df, verbose = TRUE, class_weights = 20)
    Message <rlang_message>
      epoch:   1 learn rate 0.01 	Loss: 1.3634 
      epoch:   2 learn rate 0.01 	Loss: 1.3545 
      epoch:   3 learn rate 0.01 	Loss: 1.3457 
      epoch:   4 learn rate 0.01 	Loss: 1.3369 
      epoch:   5 learn rate 0.01 	Loss: 1.3283 
      epoch:   6 learn rate 0.01 	Loss: 1.3198 
      epoch:   7 learn rate 0.01 	Loss: 1.3113 
      epoch:   8 learn rate 0.01 	Loss: 1.303 
      epoch:   9 learn rate 0.01 	Loss: 1.2947 
      epoch:  10 learn rate 0.01 	Loss: 1.2866 
      epoch:  11 learn rate 0.01 	Loss: 1.2786 
      epoch:  12 learn rate 0.01 	Loss: 1.2706 
      epoch:  13 learn rate 0.01 	Loss: 1.2627 
      epoch:  14 learn rate 0.01 	Loss: 1.2549 
      epoch:  15 learn rate 0.01 	Loss: 1.2472 
      epoch:  16 learn rate 0.01 	Loss: 1.2396 
      epoch:  17 learn rate 0.01 	Loss: 1.2321 
      epoch:  18 learn rate 0.01 	Loss: 1.2247 
      epoch:  19 learn rate 0.01 	Loss: 1.2173 
      epoch:  20 learn rate 0.01 	Loss: 1.21 
      epoch:  21 learn rate 0.01 	Loss: 1.2028 
      epoch:  22 learn rate 0.01 	Loss: 1.1957 
      epoch:  23 learn rate 0.01 	Loss: 1.1887 
      epoch:  24 learn rate 0.01 	Loss: 1.1817 
      epoch:  25 learn rate 0.01 	Loss: 1.1748 
      epoch:  26 learn rate 0.01 	Loss: 1.168 
      epoch:  27 learn rate 0.01 	Loss: 1.1612 
      epoch:  28 learn rate 0.01 	Loss: 1.1545 
      epoch:  29 learn rate 0.01 	Loss: 1.1479 
      epoch:  30 learn rate 0.01 	Loss: 1.1414 
      epoch:  31 learn rate 0.01 	Loss: 1.1349 
      epoch:  32 learn rate 0.01 	Loss: 1.1285 
      epoch:  33 learn rate 0.01 	Loss: 1.1222 
      epoch:  34 learn rate 0.01 	Loss: 1.1159 
      epoch:  35 learn rate 0.01 	Loss: 1.1097 
      epoch:  36 learn rate 0.01 	Loss: 1.1036 
      epoch:  37 learn rate 0.01 	Loss: 1.0975 
      epoch:  38 learn rate 0.01 	Loss: 1.0915 
      epoch:  39 learn rate 0.01 	Loss: 1.0855 
      epoch:  40 learn rate 0.01 	Loss: 1.0797 
      epoch:  41 learn rate 0.01 	Loss: 1.0738 
      epoch:  42 learn rate 0.01 	Loss: 1.0681 
      epoch:  43 learn rate 0.01 	Loss: 1.0624 
      epoch:  44 learn rate 0.01 	Loss: 1.0567 
      epoch:  45 learn rate 0.01 	Loss: 1.0511 
      epoch:  46 learn rate 0.01 	Loss: 1.0456 
      epoch:  47 learn rate 0.01 	Loss: 1.0401 
      epoch:  48 learn rate 0.01 	Loss: 1.0347 
      epoch:  49 learn rate 0.01 	Loss: 1.0293 
      epoch:  50 learn rate 0.01 	Loss: 1.024 
      epoch:  51 learn rate 0.01 	Loss: 1.0187 
      epoch:  52 learn rate 0.01 	Loss: 1.0135 
      epoch:  53 learn rate 0.01 	Loss: 1.0083 
      epoch:  54 learn rate 0.01 	Loss: 1.0032 
      epoch:  55 learn rate 0.01 	Loss: 0.99815 
      epoch:  56 learn rate 0.01 	Loss: 0.99315 
      epoch:  57 learn rate 0.01 	Loss: 0.98819 
      epoch:  58 learn rate 0.01 	Loss: 0.98328 
      epoch:  59 learn rate 0.01 	Loss: 0.97842 
      epoch:  60 learn rate 0.01 	Loss: 0.97361 
      epoch:  61 learn rate 0.01 	Loss: 0.96884 
      epoch:  62 learn rate 0.01 	Loss: 0.96412 
      epoch:  63 learn rate 0.01 	Loss: 0.95945 
      epoch:  64 learn rate 0.01 	Loss: 0.95483 
      epoch:  65 learn rate 0.01 	Loss: 0.95024 
      epoch:  66 learn rate 0.01 	Loss: 0.94571 
      epoch:  67 learn rate 0.01 	Loss: 0.94122 
      epoch:  68 learn rate 0.01 	Loss: 0.93677 
      epoch:  69 learn rate 0.01 	Loss: 0.93237 
      epoch:  70 learn rate 0.01 	Loss: 0.92801 
      epoch:  71 learn rate 0.01 	Loss: 0.92369 
      epoch:  72 learn rate 0.01 	Loss: 0.91942 
      epoch:  73 learn rate 0.01 	Loss: 0.91518 
      epoch:  74 learn rate 0.01 	Loss: 0.91099 
      epoch:  75 learn rate 0.01 	Loss: 0.90684 
      epoch:  76 learn rate 0.01 	Loss: 0.90273 
      epoch:  77 learn rate 0.01 	Loss: 0.89866 
      epoch:  78 learn rate 0.01 	Loss: 0.89462 
      epoch:  79 learn rate 0.01 	Loss: 0.89063 
      epoch:  80 learn rate 0.01 	Loss: 0.88667 
      epoch:  81 learn rate 0.01 	Loss: 0.88276 
      epoch:  82 learn rate 0.01 	Loss: 0.87888 
      epoch:  83 learn rate 0.01 	Loss: 0.87504 
      epoch:  84 learn rate 0.01 	Loss: 0.87123 
      epoch:  85 learn rate 0.01 	Loss: 0.86746 
      epoch:  86 learn rate 0.01 	Loss: 0.86373 
      epoch:  87 learn rate 0.01 	Loss: 0.86004 
      epoch:  88 learn rate 0.01 	Loss: 0.85637 
      epoch:  89 learn rate 0.01 	Loss: 0.85275 
      epoch:  90 learn rate 0.01 	Loss: 0.84916 
      epoch:  91 learn rate 0.01 	Loss: 0.8456 
      epoch:  92 learn rate 0.01 	Loss: 0.84208 
      epoch:  93 learn rate 0.01 	Loss: 0.83859 
      epoch:  94 learn rate 0.01 	Loss: 0.83514 
      epoch:  95 learn rate 0.01 	Loss: 0.83171 
      epoch:  96 learn rate 0.01 	Loss: 0.82832 
      epoch:  97 learn rate 0.01 	Loss: 0.82497 
      epoch:  98 learn rate 0.01 	Loss: 0.82164 
      epoch:  99 learn rate 0.01 	Loss: 0.81835 
      epoch: 100 learn rate 0.01 	Loss: 0.81509 

---

    Code
      set.seed(1)
      fit <- brulee_mlp(y ~ ., df, epochs = 2, verbose = TRUE, class_weights = c(a = 12,
        b = 1, c = 1))
    Message <rlang_message>
      epoch: 1 learn rate 0.01 	Loss: 0.72788 
      epoch: 2 learn rate 0.01 	Loss: 0.72504 

