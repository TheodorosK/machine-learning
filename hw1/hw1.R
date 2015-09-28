rm(list = ls())

library(ggplot2)
library(kknn)
library(cvTools)

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

outRMSE <- sapply(2:100, function(k) {
  print(k)
  kn <- kknn(price ~ mileage, train = cars.train, test = cars.test, k = k)
  rmse <- sqrt(mean((cars.test$price - kn$fitted.values)^2))
  return(rmse)
})

plot(outRMSE)
kMin <- which.min(outRMSE)

# TODO: n-fold cross validation (would be the "right" way to do this)
# code should get cleaned up, variable names follow no convetion, plots are ugly as hell

# This is the packaged way to do it, but the documentation is terrible
# "cv.kknn performs k-fold crossvalidation and is generally slower and does not
# yet contain the test of different models yet"
kn.cv <- cv.kknn(price ~ mileage, data = cars, kcv = 10)

# Home-made way to do k-fold cross validation

nfolds <- 10

folds <- cvFolds(n = nrow(cars), K = nfolds)

knn.cverr <- sapply(1:100, function(k) {

  print(k)
  
  # calculate CV error
  cvErr <- sapply(1:nfolds, function(f) {
    
    testInd <- folds$subsets[folds$which == f]
    trainInd <- folds$subsets[folds$which != f]
    
    cars.test <- cars[testInd, ]
    cars.train <- cars[trainInd, ]
    
    kn <- kknn(price ~ mileage, train = cars.train, test = cars.test, k = k)
    rmse <- sqrt(mean((cars.test$price - kn$fitted.values)^2))
    return(rmse)
  })
  
  return(mean(cvErr))
  
})

# TODO: should get cleaned up and put in ggplot2
plot(outRMSE)
plot(knn.cverr)

which.min(knn.cverr) # k = 40 --> minimum cross-validated error
