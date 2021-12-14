# multinomial mlp

    Code
      set.seed(1)
      fit <- brulee_mlp(y ~ ., df, epochs = 2, verbose = TRUE)
    Message <message>
      epoch: 1 	Loss: 1.1422 
      epoch: 2 	Loss: 1.1416 

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
      validation loss after 2 epochs: 1.1416 

# class weights - mlp

    Code
      set.seed(1)
      fit_imbal <- brulee_mlp(y ~ ., df, verbose = TRUE, class_weights = 20,
      optimizer = "SGD")
    Message <message>
      epoch:   1 	Loss: 1.3621 
      epoch:   2 	Loss: 1.3532 
      epoch:   3 	Loss: 1.3444 
      epoch:   4 	Loss: 1.3356 
      epoch:   5 	Loss: 1.327 
      epoch:   6 	Loss: 1.3185 
      epoch:   7 	Loss: 1.31 
      epoch:   8 	Loss: 1.3017 
      epoch:   9 	Loss: 1.2935 
      epoch:  10 	Loss: 1.2853 
      epoch:  11 	Loss: 1.2773 
      epoch:  12 	Loss: 1.2693 
      epoch:  13 	Loss: 1.2614 
      epoch:  14 	Loss: 1.2537 
      epoch:  15 	Loss: 1.246 
      epoch:  16 	Loss: 1.2384 
      epoch:  17 	Loss: 1.2308 
      epoch:  18 	Loss: 1.2234 
      epoch:  19 	Loss: 1.216 
      epoch:  20 	Loss: 1.2088 
      epoch:  21 	Loss: 1.2016 
      epoch:  22 	Loss: 1.1944 
      epoch:  23 	Loss: 1.1874 
      epoch:  24 	Loss: 1.1804 
      epoch:  25 	Loss: 1.1735 
      epoch:  26 	Loss: 1.1667 
      epoch:  27 	Loss: 1.16 
      epoch:  28 	Loss: 1.1533 
      epoch:  29 	Loss: 1.1467 
      epoch:  30 	Loss: 1.1401 
      epoch:  31 	Loss: 1.1337 
      epoch:  32 	Loss: 1.1273 
      epoch:  33 	Loss: 1.1209 
      epoch:  34 	Loss: 1.1147 
      epoch:  35 	Loss: 1.1085 
      epoch:  36 	Loss: 1.1023 
      epoch:  37 	Loss: 1.0963 
      epoch:  38 	Loss: 1.0903 
      epoch:  39 	Loss: 1.0843 
      epoch:  40 	Loss: 1.0784 
      epoch:  41 	Loss: 1.0726 
      epoch:  42 	Loss: 1.0668 
      epoch:  43 	Loss: 1.0611 
      epoch:  44 	Loss: 1.0555 
      epoch:  45 	Loss: 1.0499 
      epoch:  46 	Loss: 1.0443 
      epoch:  47 	Loss: 1.0389 
      epoch:  48 	Loss: 1.0334 
      epoch:  49 	Loss: 1.0281 
      epoch:  50 	Loss: 1.0227 
      epoch:  51 	Loss: 1.0175 
      epoch:  52 	Loss: 1.0123 
      epoch:  53 	Loss: 1.0071 
      epoch:  54 	Loss: 1.002 
      epoch:  55 	Loss: 0.99694 
      epoch:  56 	Loss: 0.99194 
      epoch:  57 	Loss: 0.98698 
      epoch:  58 	Loss: 0.98207 
      epoch:  59 	Loss: 0.97721 
      epoch:  60 	Loss: 0.9724 
      epoch:  61 	Loss: 0.96763 
      epoch:  62 	Loss: 0.96292 
      epoch:  63 	Loss: 0.95825 
      epoch:  64 	Loss: 0.95362 
      epoch:  65 	Loss: 0.94904 
      epoch:  66 	Loss: 0.94451 
      epoch:  67 	Loss: 0.94002 
      epoch:  68 	Loss: 0.93557 
      epoch:  69 	Loss: 0.93117 
      epoch:  70 	Loss: 0.92681 
      epoch:  71 	Loss: 0.92249 
      epoch:  72 	Loss: 0.91822 
      epoch:  73 	Loss: 0.91399 
      epoch:  74 	Loss: 0.9098 
      epoch:  75 	Loss: 0.90565 
      epoch:  76 	Loss: 0.90154 
      epoch:  77 	Loss: 0.89746 
      epoch:  78 	Loss: 0.89343 
      epoch:  79 	Loss: 0.88944 
      epoch:  80 	Loss: 0.88548 
      epoch:  81 	Loss: 0.88157 
      epoch:  82 	Loss: 0.87769 
      epoch:  83 	Loss: 0.87385 
      epoch:  84 	Loss: 0.87004 
      epoch:  85 	Loss: 0.86628 
      epoch:  86 	Loss: 0.86254 
      epoch:  87 	Loss: 0.85885 
      epoch:  88 	Loss: 0.85519 
      epoch:  89 	Loss: 0.85156 
      epoch:  90 	Loss: 0.84797 
      epoch:  91 	Loss: 0.84442 
      epoch:  92 	Loss: 0.84089 
      epoch:  93 	Loss: 0.83741 
      epoch:  94 	Loss: 0.83395 
      epoch:  95 	Loss: 0.83053 
      epoch:  96 	Loss: 0.82714 
      epoch:  97 	Loss: 0.82378 
      epoch:  98 	Loss: 0.82046 
      epoch:  99 	Loss: 0.81716 
      epoch: 100 	Loss: 0.8139 

---

    Code
      set.seed(1)
      fit <- brulee_mlp(y ~ ., df, epochs = 2, verbose = TRUE, class_weights = c(a = 12,
        b = 1, c = 1))
    Message <message>
      epoch: 1 	Loss: 0.72659 
      epoch: 2 	Loss: 0.72375 

