# different fit interfaces

    Code
      fit_mat
    Output
      Multilayer perceptron
      
      relu activation
      c(3) hidden units, 17 model parameters
      1,000 samples, 2 features, 2 classes 
      class weights bad=1, good=1 
      weight decay: 0.001 
      dropout proportion: 0 
      batch size: 900 
      validation cross_entropy after 10 epochs: 0.542 

# class weights - mlp

    Code
      set.seed(1)
      fit_imbal <- lantern_mlp(y ~ ., df_imbal, verbose = TRUE, class_weights = 20)
    Message <message>
      epoch:   1 cross_entropy 0.576 	 
      epoch:   2 cross_entropy 0.573 	 
      epoch:   3 cross_entropy 0.57 	 
      epoch:   4 cross_entropy 0.567 	 
      epoch:   5 cross_entropy 0.564 	 
      epoch:   6 cross_entropy 0.562 	 
      epoch:   7 cross_entropy 0.559 	 
      epoch:   8 cross_entropy 0.556 	 
      epoch:   9 cross_entropy 0.553 	 
      epoch:  10 cross_entropy 0.551 	 
      epoch:  11 cross_entropy 0.548 	 
      epoch:  12 cross_entropy 0.546 	 
      epoch:  13 cross_entropy 0.543 	 
      epoch:  14 cross_entropy 0.541 	 
      epoch:  15 cross_entropy 0.539 	 
      epoch:  16 cross_entropy 0.536 	 
      epoch:  17 cross_entropy 0.534 	 
      epoch:  18 cross_entropy 0.532 	 
      epoch:  19 cross_entropy 0.53 	 
      epoch:  20 cross_entropy 0.528 	 
      epoch:  21 cross_entropy 0.526 	 
      epoch:  22 cross_entropy 0.524 	 
      epoch:  23 cross_entropy 0.522 	 
      epoch:  24 cross_entropy 0.52 	 
      epoch:  25 cross_entropy 0.518 	 
      epoch:  26 cross_entropy 0.516 	 
      epoch:  27 cross_entropy 0.514 	 
      epoch:  28 cross_entropy 0.512 	 
      epoch:  29 cross_entropy 0.51 	 
      epoch:  30 cross_entropy 0.509 	 
      epoch:  31 cross_entropy 0.507 	 
      epoch:  32 cross_entropy 0.505 	 
      epoch:  33 cross_entropy 0.504 	 
      epoch:  34 cross_entropy 0.502 	 
      epoch:  35 cross_entropy 0.501 	 
      epoch:  36 cross_entropy 0.499 	 
      epoch:  37 cross_entropy 0.498 	 
      epoch:  38 cross_entropy 0.496 	 
      epoch:  39 cross_entropy 0.495 	 
      epoch:  40 cross_entropy 0.493 	 
      epoch:  41 cross_entropy 0.492 	 
      epoch:  42 cross_entropy 0.49 	 
      epoch:  43 cross_entropy 0.489 	 
      epoch:  44 cross_entropy 0.488 	 
      epoch:  45 cross_entropy 0.486 	 
      epoch:  46 cross_entropy 0.485 	 
      epoch:  47 cross_entropy 0.484 	 
      epoch:  48 cross_entropy 0.483 	 
      epoch:  49 cross_entropy 0.481 	 
      epoch:  50 cross_entropy 0.48 	 
      epoch:  51 cross_entropy 0.479 	 
      epoch:  52 cross_entropy 0.478 	 
      epoch:  53 cross_entropy 0.477 	 
      epoch:  54 cross_entropy 0.476 	 
      epoch:  55 cross_entropy 0.474 	 
      epoch:  56 cross_entropy 0.473 	 
      epoch:  57 cross_entropy 0.472 	 
      epoch:  58 cross_entropy 0.471 	 
      epoch:  59 cross_entropy 0.47 	 
      epoch:  60 cross_entropy 0.469 	 
      epoch:  61 cross_entropy 0.468 	 
      epoch:  62 cross_entropy 0.467 	 
      epoch:  63 cross_entropy 0.466 	 
      epoch:  64 cross_entropy 0.465 	 
      epoch:  65 cross_entropy 0.464 	 
      epoch:  66 cross_entropy 0.463 	 
      epoch:  67 cross_entropy 0.462 	 
      epoch:  68 cross_entropy 0.461 	 
      epoch:  69 cross_entropy 0.46 	 
      epoch:  70 cross_entropy 0.46 	 
      epoch:  71 cross_entropy 0.459 	 
      epoch:  72 cross_entropy 0.458 	 
      epoch:  73 cross_entropy 0.457 	 
      epoch:  74 cross_entropy 0.456 	 
      epoch:  75 cross_entropy 0.455 	 
      epoch:  76 cross_entropy 0.454 	 
      epoch:  77 cross_entropy 0.454 	 
      epoch:  78 cross_entropy 0.453 	 
      epoch:  79 cross_entropy 0.452 	 
      epoch:  80 cross_entropy 0.451 	 
      epoch:  81 cross_entropy 0.45 	 
      epoch:  82 cross_entropy 0.45 	 
      epoch:  83 cross_entropy 0.449 	 
      epoch:  84 cross_entropy 0.448 	 
      epoch:  85 cross_entropy 0.447 	 
      epoch:  86 cross_entropy 0.447 	 
      epoch:  87 cross_entropy 0.446 	 
      epoch:  88 cross_entropy 0.445 	 
      epoch:  89 cross_entropy 0.445 	 
      epoch:  90 cross_entropy 0.444 	 
      epoch:  91 cross_entropy 0.443 	 
      epoch:  92 cross_entropy 0.442 	 
      epoch:  93 cross_entropy 0.442 	 
      epoch:  94 cross_entropy 0.441 	 
      epoch:  95 cross_entropy 0.44 	 
      epoch:  96 cross_entropy 0.44 	 
      epoch:  97 cross_entropy 0.439 	 
      epoch:  98 cross_entropy 0.438 	 
      epoch:  99 cross_entropy 0.438 	 
      epoch: 100 cross_entropy 0.437 	 

---

    Code
      set.seed(1)
      fit <- lantern_mlp(y ~ ., df_imbal, epochs = 2, verbose = TRUE, class_weights = c(
        a = 12, b = 1))
    Message <message>
      epoch: 1 cross_entropy 0.665 	 
      epoch: 2 cross_entropy 0.663 	 

