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
      learn rate: 0.01 
      validation loss after 10 epochs: 0.543 

# class weights - mlp

    Code
      set.seed(1)
      fit_imbal <- brulee_mlp(y ~ ., df_imbal, verbose = TRUE, class_weights = 20)
    Message <rlang_message>
      epoch:   1 learn rate 0.01 	Loss: 0.578 
      epoch:   2 learn rate 0.01 	Loss: 0.574 
      epoch:   3 learn rate 0.01 	Loss: 0.571 
      epoch:   4 learn rate 0.01 	Loss: 0.568 
      epoch:   5 learn rate 0.01 	Loss: 0.565 
      epoch:   6 learn rate 0.01 	Loss: 0.563 
      epoch:   7 learn rate 0.01 	Loss: 0.56 
      epoch:   8 learn rate 0.01 	Loss: 0.557 
      epoch:   9 learn rate 0.01 	Loss: 0.554 
      epoch:  10 learn rate 0.01 	Loss: 0.552 
      epoch:  11 learn rate 0.01 	Loss: 0.549 
      epoch:  12 learn rate 0.01 	Loss: 0.547 
      epoch:  13 learn rate 0.01 	Loss: 0.544 
      epoch:  14 learn rate 0.01 	Loss: 0.542 
      epoch:  15 learn rate 0.01 	Loss: 0.54 
      epoch:  16 learn rate 0.01 	Loss: 0.537 
      epoch:  17 learn rate 0.01 	Loss: 0.535 
      epoch:  18 learn rate 0.01 	Loss: 0.533 
      epoch:  19 learn rate 0.01 	Loss: 0.531 
      epoch:  20 learn rate 0.01 	Loss: 0.529 
      epoch:  21 learn rate 0.01 	Loss: 0.527 
      epoch:  22 learn rate 0.01 	Loss: 0.525 
      epoch:  23 learn rate 0.01 	Loss: 0.523 
      epoch:  24 learn rate 0.01 	Loss: 0.521 
      epoch:  25 learn rate 0.01 	Loss: 0.519 
      epoch:  26 learn rate 0.01 	Loss: 0.517 
      epoch:  27 learn rate 0.01 	Loss: 0.515 
      epoch:  28 learn rate 0.01 	Loss: 0.513 
      epoch:  29 learn rate 0.01 	Loss: 0.511 
      epoch:  30 learn rate 0.01 	Loss: 0.51 
      epoch:  31 learn rate 0.01 	Loss: 0.508 
      epoch:  32 learn rate 0.01 	Loss: 0.506 
      epoch:  33 learn rate 0.01 	Loss: 0.505 
      epoch:  34 learn rate 0.01 	Loss: 0.503 
      epoch:  35 learn rate 0.01 	Loss: 0.502 
      epoch:  36 learn rate 0.01 	Loss: 0.5 
      epoch:  37 learn rate 0.01 	Loss: 0.499 
      epoch:  38 learn rate 0.01 	Loss: 0.497 
      epoch:  39 learn rate 0.01 	Loss: 0.496 
      epoch:  40 learn rate 0.01 	Loss: 0.494 
      epoch:  41 learn rate 0.01 	Loss: 0.493 
      epoch:  42 learn rate 0.01 	Loss: 0.491 
      epoch:  43 learn rate 0.01 	Loss: 0.49 
      epoch:  44 learn rate 0.01 	Loss: 0.489 
      epoch:  45 learn rate 0.01 	Loss: 0.487 
      epoch:  46 learn rate 0.01 	Loss: 0.486 
      epoch:  47 learn rate 0.01 	Loss: 0.485 
      epoch:  48 learn rate 0.01 	Loss: 0.484 
      epoch:  49 learn rate 0.01 	Loss: 0.482 
      epoch:  50 learn rate 0.01 	Loss: 0.481 
      epoch:  51 learn rate 0.01 	Loss: 0.48 
      epoch:  52 learn rate 0.01 	Loss: 0.479 
      epoch:  53 learn rate 0.01 	Loss: 0.478 
      epoch:  54 learn rate 0.01 	Loss: 0.477 
      epoch:  55 learn rate 0.01 	Loss: 0.475 
      epoch:  56 learn rate 0.01 	Loss: 0.474 
      epoch:  57 learn rate 0.01 	Loss: 0.473 
      epoch:  58 learn rate 0.01 	Loss: 0.472 
      epoch:  59 learn rate 0.01 	Loss: 0.471 
      epoch:  60 learn rate 0.01 	Loss: 0.47 
      epoch:  61 learn rate 0.01 	Loss: 0.469 
      epoch:  62 learn rate 0.01 	Loss: 0.468 
      epoch:  63 learn rate 0.01 	Loss: 0.467 
      epoch:  64 learn rate 0.01 	Loss: 0.466 
      epoch:  65 learn rate 0.01 	Loss: 0.465 
      epoch:  66 learn rate 0.01 	Loss: 0.464 
      epoch:  67 learn rate 0.01 	Loss: 0.463 
      epoch:  68 learn rate 0.01 	Loss: 0.462 
      epoch:  69 learn rate 0.01 	Loss: 0.461 
      epoch:  70 learn rate 0.01 	Loss: 0.461 
      epoch:  71 learn rate 0.01 	Loss: 0.46 
      epoch:  72 learn rate 0.01 	Loss: 0.459 
      epoch:  73 learn rate 0.01 	Loss: 0.458 
      epoch:  74 learn rate 0.01 	Loss: 0.457 
      epoch:  75 learn rate 0.01 	Loss: 0.456 
      epoch:  76 learn rate 0.01 	Loss: 0.455 
      epoch:  77 learn rate 0.01 	Loss: 0.455 
      epoch:  78 learn rate 0.01 	Loss: 0.454 
      epoch:  79 learn rate 0.01 	Loss: 0.453 
      epoch:  80 learn rate 0.01 	Loss: 0.452 
      epoch:  81 learn rate 0.01 	Loss: 0.451 
      epoch:  82 learn rate 0.01 	Loss: 0.451 
      epoch:  83 learn rate 0.01 	Loss: 0.45 
      epoch:  84 learn rate 0.01 	Loss: 0.449 
      epoch:  85 learn rate 0.01 	Loss: 0.448 
      epoch:  86 learn rate 0.01 	Loss: 0.448 
      epoch:  87 learn rate 0.01 	Loss: 0.447 
      epoch:  88 learn rate 0.01 	Loss: 0.446 
      epoch:  89 learn rate 0.01 	Loss: 0.446 
      epoch:  90 learn rate 0.01 	Loss: 0.445 
      epoch:  91 learn rate 0.01 	Loss: 0.444 
      epoch:  92 learn rate 0.01 	Loss: 0.443 
      epoch:  93 learn rate 0.01 	Loss: 0.443 
      epoch:  94 learn rate 0.01 	Loss: 0.442 
      epoch:  95 learn rate 0.01 	Loss: 0.441 
      epoch:  96 learn rate 0.01 	Loss: 0.441 
      epoch:  97 learn rate 0.01 	Loss: 0.44 
      epoch:  98 learn rate 0.01 	Loss: 0.44 
      epoch:  99 learn rate 0.01 	Loss: 0.439 
      epoch: 100 learn rate 0.01 	Loss: 0.438 

---

    Code
      set.seed(1)
      fit <- brulee_mlp(y ~ ., df_imbal, epochs = 2, verbose = TRUE, class_weights = c(
        a = 12, b = 1))
    Message <rlang_message>
      epoch: 1 learn rate 0.01 	Loss: 0.666 
      epoch: 2 learn rate 0.01 	Loss: 0.664 

