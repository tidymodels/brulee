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
      validation loss after 10 epochs: 0.542 

# class weights - mlp

    Code
      set.seed(1)
      fit_imbal <- lantern_mlp(y ~ ., df_imbal, verbose = TRUE, class_weights = 20)
    Message <message>
      epoch:   1 	Loss: 0.576 
      epoch:   2 	Loss: 0.573 
      epoch:   3 	Loss: 0.57 
      epoch:   4 	Loss: 0.567 
      epoch:   5 	Loss: 0.564 
      epoch:   6 	Loss: 0.562 
      epoch:   7 	Loss: 0.559 
      epoch:   8 	Loss: 0.556 
      epoch:   9 	Loss: 0.553 
      epoch:  10 	Loss: 0.551 
      epoch:  11 	Loss: 0.548 
      epoch:  12 	Loss: 0.546 
      epoch:  13 	Loss: 0.543 
      epoch:  14 	Loss: 0.541 
      epoch:  15 	Loss: 0.539 
      epoch:  16 	Loss: 0.536 
      epoch:  17 	Loss: 0.534 
      epoch:  18 	Loss: 0.532 
      epoch:  19 	Loss: 0.53 
      epoch:  20 	Loss: 0.528 
      epoch:  21 	Loss: 0.526 
      epoch:  22 	Loss: 0.524 
      epoch:  23 	Loss: 0.522 
      epoch:  24 	Loss: 0.52 
      epoch:  25 	Loss: 0.518 
      epoch:  26 	Loss: 0.516 
      epoch:  27 	Loss: 0.514 
      epoch:  28 	Loss: 0.512 
      epoch:  29 	Loss: 0.51 
      epoch:  30 	Loss: 0.509 
      epoch:  31 	Loss: 0.507 
      epoch:  32 	Loss: 0.505 
      epoch:  33 	Loss: 0.504 
      epoch:  34 	Loss: 0.502 
      epoch:  35 	Loss: 0.501 
      epoch:  36 	Loss: 0.499 
      epoch:  37 	Loss: 0.498 
      epoch:  38 	Loss: 0.496 
      epoch:  39 	Loss: 0.495 
      epoch:  40 	Loss: 0.493 
      epoch:  41 	Loss: 0.492 
      epoch:  42 	Loss: 0.49 
      epoch:  43 	Loss: 0.489 
      epoch:  44 	Loss: 0.488 
      epoch:  45 	Loss: 0.486 
      epoch:  46 	Loss: 0.485 
      epoch:  47 	Loss: 0.484 
      epoch:  48 	Loss: 0.483 
      epoch:  49 	Loss: 0.481 
      epoch:  50 	Loss: 0.48 
      epoch:  51 	Loss: 0.479 
      epoch:  52 	Loss: 0.478 
      epoch:  53 	Loss: 0.477 
      epoch:  54 	Loss: 0.476 
      epoch:  55 	Loss: 0.474 
      epoch:  56 	Loss: 0.473 
      epoch:  57 	Loss: 0.472 
      epoch:  58 	Loss: 0.471 
      epoch:  59 	Loss: 0.47 
      epoch:  60 	Loss: 0.469 
      epoch:  61 	Loss: 0.468 
      epoch:  62 	Loss: 0.467 
      epoch:  63 	Loss: 0.466 
      epoch:  64 	Loss: 0.465 
      epoch:  65 	Loss: 0.464 
      epoch:  66 	Loss: 0.463 
      epoch:  67 	Loss: 0.462 
      epoch:  68 	Loss: 0.461 
      epoch:  69 	Loss: 0.46 
      epoch:  70 	Loss: 0.46 
      epoch:  71 	Loss: 0.459 
      epoch:  72 	Loss: 0.458 
      epoch:  73 	Loss: 0.457 
      epoch:  74 	Loss: 0.456 
      epoch:  75 	Loss: 0.455 
      epoch:  76 	Loss: 0.454 
      epoch:  77 	Loss: 0.454 
      epoch:  78 	Loss: 0.453 
      epoch:  79 	Loss: 0.452 
      epoch:  80 	Loss: 0.451 
      epoch:  81 	Loss: 0.45 
      epoch:  82 	Loss: 0.45 
      epoch:  83 	Loss: 0.449 
      epoch:  84 	Loss: 0.448 
      epoch:  85 	Loss: 0.447 
      epoch:  86 	Loss: 0.447 
      epoch:  87 	Loss: 0.446 
      epoch:  88 	Loss: 0.445 
      epoch:  89 	Loss: 0.445 
      epoch:  90 	Loss: 0.444 
      epoch:  91 	Loss: 0.443 
      epoch:  92 	Loss: 0.442 
      epoch:  93 	Loss: 0.442 
      epoch:  94 	Loss: 0.441 
      epoch:  95 	Loss: 0.44 
      epoch:  96 	Loss: 0.44 
      epoch:  97 	Loss: 0.439 
      epoch:  98 	Loss: 0.438 
      epoch:  99 	Loss: 0.438 
      epoch: 100 	Loss: 0.437 

---

    Code
      set.seed(1)
      fit <- lantern_mlp(y ~ ., df_imbal, epochs = 2, verbose = TRUE, class_weights = c(
        a = 12, b = 1))
    Message <message>
      epoch: 1 	Loss: 0.665 
      epoch: 2 	Loss: 0.663 

