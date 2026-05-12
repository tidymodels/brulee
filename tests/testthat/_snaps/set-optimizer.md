# set_optimizer returns correct optimizer types

    Code
      brulee:::set_optimizer("Invalid", model, 0.1, 0.9, 0.01, 0)
    Condition
      Error in `brulee:::set_optimizer()`:
      ! Unsupported optimizer 'Invalid'

