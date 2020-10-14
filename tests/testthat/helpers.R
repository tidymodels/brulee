library(modeldata)

## -----------------------------------------------------------------------------

data(ames)
ames$Sale_Price <- log10(ames$Sale_Price)

ames_x_df <- ames[, c("Longitude", "Latitude")]
ames_x_mat <- as.matrix(ames_x_df)
ames_y <- ames$Sale_Price
ames_rec <- recipe(Sale_Price ~ Longitude + Latitude, data = ames)

## -----------------------------------------------------------------------------

