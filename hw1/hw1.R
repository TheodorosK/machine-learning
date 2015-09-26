rm(list = ls())

library(ggplot2)
library(kknn)
library(tictoc)
library(parallel)

set.seed(926)

# Read data

cars <- read.csv("../data/susedcars.csv")

# Visualize

g <- ggplot(data = cars, aes(x = mileage, y = price)) + geom_point()

print(g)

# Fit linear relationship

lin <- glm(mileage ~ price, data = cars)

# Fit knn

nTrain <- round(0.9 * nrow(cars))
tInd <- sample(1:nrow(cars), nTrain)

cars.train <- cars[tInd, ]
cars.test <- cars[-tInd, ]

# estimate RMSE for all possible values of k

tic()
outRMSE <- sapply(2:100, function(k) {
  print(k)
  kn <- kknn(price ~ mileage, train = cars.train, test = cars.test, k = k)
  rmse <- sqrt(mean((cars.test$price - kn$fitted.values)^2))
  return(rmse)
})
toc()

plot(outRMSE)
kMin <- which.min(outRMSE)

# TODO: n-fold cross validation (would be the "right" way to do this)