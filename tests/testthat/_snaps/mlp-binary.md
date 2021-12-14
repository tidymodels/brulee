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
      validation loss after 10 epochs: 0.54249 

# class weights - mlp

    Code
      set.seed(1)
      fit_imbal <- brulee_mlp(y ~ ., df_imbal, verbose = TRUE, class_weights = 20)
    Message <message>
      epoch:   1 	Loss: 0.57648 
      epoch:   2 	Loss: 0.57335 
      epoch:   3 	Loss: 0.5703 
      epoch:   4 	Loss: 0.56733 
      epoch:   5 	Loss: 0.56443 
      epoch:   6 	Loss: 0.56161 
      epoch:   7 	Loss: 0.55879 
      epoch:   8 	Loss: 0.55603 
      epoch:   9 	Loss: 0.55335 
      epoch:  10 	Loss: 0.55074 
      epoch:  11 	Loss: 0.54819 
      epoch:  12 	Loss: 0.54571 
      epoch:  13 	Loss: 0.54328 
      epoch:  14 	Loss: 0.54091 
      epoch:  15 	Loss: 0.53859 
      epoch:  16 	Loss: 0.53633 
      epoch:  17 	Loss: 0.5341 
      epoch:  18 	Loss: 0.53193 
      epoch:  19 	Loss: 0.5298 
      epoch:  20 	Loss: 0.52772 
      epoch:  21 	Loss: 0.52563 
      epoch:  22 	Loss: 0.52359 
      epoch:  23 	Loss: 0.52159 
      epoch:  24 	Loss: 0.51964 
      epoch:  25 	Loss: 0.51772 
      epoch:  26 	Loss: 0.51585 
      epoch:  27 	Loss: 0.51402 
      epoch:  28 	Loss: 0.51222 
      epoch:  29 	Loss: 0.51046 
      epoch:  30 	Loss: 0.50873 
      epoch:  31 	Loss: 0.50704 
      epoch:  32 	Loss: 0.50538 
      epoch:  33 	Loss: 0.50376 
      epoch:  34 	Loss: 0.50218 
      epoch:  35 	Loss: 0.50064 
      epoch:  36 	Loss: 0.49912 
      epoch:  37 	Loss: 0.49764 
      epoch:  38 	Loss: 0.49617 
      epoch:  39 	Loss: 0.49469 
      epoch:  40 	Loss: 0.49325 
      epoch:  41 	Loss: 0.49184 
      epoch:  42 	Loss: 0.49046 
      epoch:  43 	Loss: 0.4891 
      epoch:  44 	Loss: 0.48776 
      epoch:  45 	Loss: 0.48645 
      epoch:  46 	Loss: 0.48516 
      epoch:  47 	Loss: 0.48389 
      epoch:  48 	Loss: 0.48265 
      epoch:  49 	Loss: 0.48142 
      epoch:  50 	Loss: 0.48021 
      epoch:  51 	Loss: 0.47902 
      epoch:  52 	Loss: 0.47785 
      epoch:  53 	Loss: 0.4767 
      epoch:  54 	Loss: 0.47557 
      epoch:  55 	Loss: 0.47445 
      epoch:  56 	Loss: 0.47335 
      epoch:  57 	Loss: 0.47226 
      epoch:  58 	Loss: 0.47119 
      epoch:  59 	Loss: 0.47014 
      epoch:  60 	Loss: 0.46911 
      epoch:  61 	Loss: 0.46809 
      epoch:  62 	Loss: 0.46709 
      epoch:  63 	Loss: 0.4661 
      epoch:  64 	Loss: 0.46513 
      epoch:  65 	Loss: 0.46417 
      epoch:  66 	Loss: 0.46322 
      epoch:  67 	Loss: 0.46229 
      epoch:  68 	Loss: 0.46137 
      epoch:  69 	Loss: 0.46046 
      epoch:  70 	Loss: 0.45956 
      epoch:  71 	Loss: 0.45867 
      epoch:  72 	Loss: 0.4578 
      epoch:  73 	Loss: 0.45693 
      epoch:  74 	Loss: 0.45608 
      epoch:  75 	Loss: 0.45523 
      epoch:  76 	Loss: 0.4544 
      epoch:  77 	Loss: 0.45357 
      epoch:  78 	Loss: 0.45276 
      epoch:  79 	Loss: 0.45195 
      epoch:  80 	Loss: 0.45115 
      epoch:  81 	Loss: 0.45037 
      epoch:  82 	Loss: 0.44961 
      epoch:  83 	Loss: 0.44886 
      epoch:  84 	Loss: 0.44811 
      epoch:  85 	Loss: 0.44737 
      epoch:  86 	Loss: 0.44665 
      epoch:  87 	Loss: 0.44593 
      epoch:  88 	Loss: 0.44521 
      epoch:  89 	Loss: 0.44451 
      epoch:  90 	Loss: 0.44381 
      epoch:  91 	Loss: 0.44312 
      epoch:  92 	Loss: 0.44243 
      epoch:  93 	Loss: 0.44175 
      epoch:  94 	Loss: 0.44109 
      epoch:  95 	Loss: 0.44043 
      epoch:  96 	Loss: 0.43977 
      epoch:  97 	Loss: 0.43913 
      epoch:  98 	Loss: 0.43849 
      epoch:  99 	Loss: 0.43785 
      epoch: 100 	Loss: 0.43722 

---

    Code
      set.seed(1)
      fit <- brulee_mlp(y ~ ., df_imbal, epochs = 2, verbose = TRUE, class_weights = c(
        a = 12, b = 1))
    Message <message>
      epoch: 1 	Loss: 0.66515 
      epoch: 2 	Loss: 0.66288 

