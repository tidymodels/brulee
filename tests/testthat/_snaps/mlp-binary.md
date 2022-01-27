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
      validation loss after 10 epochs: 0.5459 

# class weights - mlp

    Code
      set.seed(1)
      fit_imbal <- brulee_mlp(y ~ ., df_imbal, verbose = TRUE, class_weights = 20)
    Message <rlang_message>
      epoch:   1 	Loss: 0.581 
      epoch:   2 	Loss: 0.57786 
      epoch:   3 	Loss: 0.5748 
      epoch:   4 	Loss: 0.57181 
      epoch:   5 	Loss: 0.56891 
      epoch:   6 	Loss: 0.56608 
      epoch:   7 	Loss: 0.56324 
      epoch:   8 	Loss: 0.56048 
      epoch:   9 	Loss: 0.55779 
      epoch:  10 	Loss: 0.55517 
      epoch:  11 	Loss: 0.55261 
      epoch:  12 	Loss: 0.55012 
      epoch:  13 	Loss: 0.54768 
      epoch:  14 	Loss: 0.5453 
      epoch:  15 	Loss: 0.54298 
      epoch:  16 	Loss: 0.54071 
      epoch:  17 	Loss: 0.53848 
      epoch:  18 	Loss: 0.53629 
      epoch:  19 	Loss: 0.53416 
      epoch:  20 	Loss: 0.53207 
      epoch:  21 	Loss: 0.52998 
      epoch:  22 	Loss: 0.52793 
      epoch:  23 	Loss: 0.52592 
      epoch:  24 	Loss: 0.52396 
      epoch:  25 	Loss: 0.52204 
      epoch:  26 	Loss: 0.52016 
      epoch:  27 	Loss: 0.51832 
      epoch:  28 	Loss: 0.51652 
      epoch:  29 	Loss: 0.51475 
      epoch:  30 	Loss: 0.51302 
      epoch:  31 	Loss: 0.51133 
      epoch:  32 	Loss: 0.50966 
      epoch:  33 	Loss: 0.50804 
      epoch:  34 	Loss: 0.50646 
      epoch:  35 	Loss: 0.50491 
      epoch:  36 	Loss: 0.50339 
      epoch:  37 	Loss: 0.5019 
      epoch:  38 	Loss: 0.50044 
      epoch:  39 	Loss: 0.49895 
      epoch:  40 	Loss: 0.49751 
      epoch:  41 	Loss: 0.49609 
      epoch:  42 	Loss: 0.49471 
      epoch:  43 	Loss: 0.49334 
      epoch:  44 	Loss: 0.49201 
      epoch:  45 	Loss: 0.49069 
      epoch:  46 	Loss: 0.4894 
      epoch:  47 	Loss: 0.48813 
      epoch:  48 	Loss: 0.48688 
      epoch:  49 	Loss: 0.48565 
      epoch:  50 	Loss: 0.48444 
      epoch:  51 	Loss: 0.48325 
      epoch:  52 	Loss: 0.48207 
      epoch:  53 	Loss: 0.48092 
      epoch:  54 	Loss: 0.47978 
      epoch:  55 	Loss: 0.47866 
      epoch:  56 	Loss: 0.47756 
      epoch:  57 	Loss: 0.47647 
      epoch:  58 	Loss: 0.4754 
      epoch:  59 	Loss: 0.47435 
      epoch:  60 	Loss: 0.47331 
      epoch:  61 	Loss: 0.47229 
      epoch:  62 	Loss: 0.47129 
      epoch:  63 	Loss: 0.4703 
      epoch:  64 	Loss: 0.46932 
      epoch:  65 	Loss: 0.46836 
      epoch:  66 	Loss: 0.46741 
      epoch:  67 	Loss: 0.46648 
      epoch:  68 	Loss: 0.46555 
      epoch:  69 	Loss: 0.46464 
      epoch:  70 	Loss: 0.46374 
      epoch:  71 	Loss: 0.46286 
      epoch:  72 	Loss: 0.46198 
      epoch:  73 	Loss: 0.46111 
      epoch:  74 	Loss: 0.46026 
      epoch:  75 	Loss: 0.45941 
      epoch:  76 	Loss: 0.45858 
      epoch:  77 	Loss: 0.45775 
      epoch:  78 	Loss: 0.45694 
      epoch:  79 	Loss: 0.45613 
      epoch:  80 	Loss: 0.45533 
      epoch:  81 	Loss: 0.45454 
      epoch:  82 	Loss: 0.45377 
      epoch:  83 	Loss: 0.45301 
      epoch:  84 	Loss: 0.45227 
      epoch:  85 	Loss: 0.45153 
      epoch:  86 	Loss: 0.4508 
      epoch:  87 	Loss: 0.45008 
      epoch:  88 	Loss: 0.44937 
      epoch:  89 	Loss: 0.44866 
      epoch:  90 	Loss: 0.44796 
      epoch:  91 	Loss: 0.44727 
      epoch:  92 	Loss: 0.44659 
      epoch:  93 	Loss: 0.44591 
      epoch:  94 	Loss: 0.44524 
      epoch:  95 	Loss: 0.44459 
      epoch:  96 	Loss: 0.44393 
      epoch:  97 	Loss: 0.44329 
      epoch:  98 	Loss: 0.44265 
      epoch:  99 	Loss: 0.44201 
      epoch: 100 	Loss: 0.44138 

---

    Code
      set.seed(1)
      fit <- brulee_mlp(y ~ ., df_imbal, epochs = 2, verbose = TRUE, class_weights = c(
        a = 12, b = 1))
    Message <rlang_message>
      epoch: 1 	Loss: 0.66968 
      epoch: 2 	Loss: 0.66739 

