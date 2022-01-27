# multinomial mlp

    Code
      set.seed(1)
      fit <- brulee_mlp(y ~ ., df, epochs = 2, verbose = TRUE)
    Message <message>
      epoch: 1 	Loss: 1.1479 
      epoch: 2 	Loss: 1.1472 

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
      validation loss after 2 epochs: 1.1472 

# class weights - mlp

    Code
      set.seed(1)
      fit_imbal <- brulee_mlp(y ~ ., df, verbose = TRUE, class_weights = 20,
      optimizer = "SGD")
    Message <message>
      epoch:   1 	Loss: 1.3678 
      epoch:   2 	Loss: 1.3588 
      epoch:   3 	Loss: 1.35 
      epoch:   4 	Loss: 1.3412 
      epoch:   5 	Loss: 1.3326 
      epoch:   6 	Loss: 1.3241 
      epoch:   7 	Loss: 1.3157 
      epoch:   8 	Loss: 1.3073 
      epoch:   9 	Loss: 1.2991 
      epoch:  10 	Loss: 1.291 
      epoch:  11 	Loss: 1.2829 
      epoch:  12 	Loss: 1.2749 
      epoch:  13 	Loss: 1.2671 
      epoch:  14 	Loss: 1.2593 
      epoch:  15 	Loss: 1.2516 
      epoch:  16 	Loss: 1.244 
      epoch:  17 	Loss: 1.2365 
      epoch:  18 	Loss: 1.229 
      epoch:  19 	Loss: 1.2217 
      epoch:  20 	Loss: 1.2144 
      epoch:  21 	Loss: 1.2072 
      epoch:  22 	Loss: 1.2001 
      epoch:  23 	Loss: 1.193 
      epoch:  24 	Loss: 1.1861 
      epoch:  25 	Loss: 1.1792 
      epoch:  26 	Loss: 1.1724 
      epoch:  27 	Loss: 1.1656 
      epoch:  28 	Loss: 1.1589 
      epoch:  29 	Loss: 1.1523 
      epoch:  30 	Loss: 1.1458 
      epoch:  31 	Loss: 1.1393 
      epoch:  32 	Loss: 1.1329 
      epoch:  33 	Loss: 1.1266 
      epoch:  34 	Loss: 1.1203 
      epoch:  35 	Loss: 1.1141 
      epoch:  36 	Loss: 1.108 
      epoch:  37 	Loss: 1.1019 
      epoch:  38 	Loss: 1.0959 
      epoch:  39 	Loss: 1.09 
      epoch:  40 	Loss: 1.0841 
      epoch:  41 	Loss: 1.0783 
      epoch:  42 	Loss: 1.0725 
      epoch:  43 	Loss: 1.0668 
      epoch:  44 	Loss: 1.0611 
      epoch:  45 	Loss: 1.0555 
      epoch:  46 	Loss: 1.05 
      epoch:  47 	Loss: 1.0445 
      epoch:  48 	Loss: 1.0391 
      epoch:  49 	Loss: 1.0337 
      epoch:  50 	Loss: 1.0284 
      epoch:  51 	Loss: 1.0232 
      epoch:  52 	Loss: 1.0179 
      epoch:  53 	Loss: 1.0128 
      epoch:  54 	Loss: 1.0077 
      epoch:  55 	Loss: 1.0026 
      epoch:  56 	Loss: 0.99762 
      epoch:  57 	Loss: 0.99266 
      epoch:  58 	Loss: 0.98775 
      epoch:  59 	Loss: 0.9829 
      epoch:  60 	Loss: 0.97808 
      epoch:  61 	Loss: 0.97332 
      epoch:  62 	Loss: 0.96861 
      epoch:  63 	Loss: 0.96394 
      epoch:  64 	Loss: 0.95931 
      epoch:  65 	Loss: 0.95473 
      epoch:  66 	Loss: 0.9502 
      epoch:  67 	Loss: 0.94571 
      epoch:  68 	Loss: 0.94127 
      epoch:  69 	Loss: 0.93687 
      epoch:  70 	Loss: 0.93251 
      epoch:  71 	Loss: 0.92819 
      epoch:  72 	Loss: 0.92392 
      epoch:  73 	Loss: 0.91969 
      epoch:  74 	Loss: 0.9155 
      epoch:  75 	Loss: 0.91135 
      epoch:  76 	Loss: 0.90724 
      epoch:  77 	Loss: 0.90317 
      epoch:  78 	Loss: 0.89914 
      epoch:  79 	Loss: 0.89514 
      epoch:  80 	Loss: 0.89119 
      epoch:  81 	Loss: 0.88728 
      epoch:  82 	Loss: 0.8834 
      epoch:  83 	Loss: 0.87956 
      epoch:  84 	Loss: 0.87575 
      epoch:  85 	Loss: 0.87199 
      epoch:  86 	Loss: 0.86826 
      epoch:  87 	Loss: 0.86456 
      epoch:  88 	Loss: 0.8609 
      epoch:  89 	Loss: 0.85728 
      epoch:  90 	Loss: 0.85369 
      epoch:  91 	Loss: 0.85013 
      epoch:  92 	Loss: 0.84661 
      epoch:  93 	Loss: 0.84313 
      epoch:  94 	Loss: 0.83967 
      epoch:  95 	Loss: 0.83625 
      epoch:  96 	Loss: 0.83286 
      epoch:  97 	Loss: 0.82951 
      epoch:  98 	Loss: 0.82618 
      epoch:  99 	Loss: 0.82289 
      epoch: 100 	Loss: 0.81963 

---

    Code
      set.seed(1)
      fit <- brulee_mlp(y ~ ., df, epochs = 2, verbose = TRUE, class_weights = c(a = 12,
        b = 1, c = 1))
    Message <message>
      epoch: 1 	Loss: 0.73219 
      epoch: 2 	Loss: 0.72935 

