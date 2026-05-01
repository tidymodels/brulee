# Test script to verify CUDA device support works correctly
# Run this on a machine with CUDA available

library(brulee)
library(modeldata)

# Check if CUDA is available
if (!torch::cuda_is_available()) {
  stop("CUDA is not available on this system. This test requires CUDA.")
}

cat("CUDA is available. Testing device support...\n\n")

# Test 1: Regression with MLP on CUDA
cat("Test 1: MLP regression on CUDA\n")
set.seed(1)
dat <- sim_regression(200, method = "sapp_2014_1")

system.time({
  set.seed(2)
  fit_gpu <- brulee_mlp(
    outcome ~ .,
    data = dat,
    epochs = 50,
    stop_iter = 5,
    hidden_units = 50,
    device = "cuda",
    verbose = FALSE
  )
})

cat("✓ MLP trained on CUDA successfully\n")
cat("Device:", fit_gpu$device, "\n")

# Test prediction
pred <- predict(fit_gpu, dat[1:5, ])
cat("✓ Prediction works\n\n")

# Test 2: Classification with class weights on CUDA
cat("Test 2: Logistic regression with class weights on CUDA\n")
data(parabolic)

set.seed(3)
fit_cls <- brulee_logistic_reg(
  class ~ .,
  data = parabolic[1:200, ],
  epochs = 20,
  device = "cuda",
  class_weights = c(Class1 = 2, Class2 = 1),
  verbose = FALSE
)

cat("✓ Classification with class_weights trained on CUDA successfully\n")
cat("Device:", fit_cls$device, "\n")

pred_cls <- predict(fit_cls, parabolic[201:210, ], type = "prob")
cat("✓ Prediction works\n\n")

# Test 3: ResNet on CUDA
cat("Test 3: ResNet on CUDA\n")
set.seed(4)
fit_resnet <- brulee_resnet(
  outcome ~ .,
  data = dat,
  hidden_units = c(10, 5),
  batch_norm_units = c(8, 4),
  residual_at = 2,
  epochs = 10,
  device = "cuda",
  verbose = FALSE
)

cat("✓ ResNet trained on CUDA successfully\n")
cat("Device:", fit_resnet$device, "\n\n")

cat("All tests passed! ✓\n")
