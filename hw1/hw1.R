##### HW1
#
# Setup ----
#
rm(list = ls())

library(ggplot2)
library(kknn)
library(cvTools)
library(reshape2)

set.seed(926)

#
# Read data ----
#
cars <- read.csv("../data/susedcars.csv")

# Visualize
lin <- glm(price ~ mileage, data = cars)

# # Reformat linear coefficients for plotting (there's probably an easier way)
# # mileage=mx+b
# # 1/m*Y +(-b/m) = X
# # (y-b)/m = x
# linCoefs <- data.frame("intercept" = -lin$coefficients[1]/lin$coefficients[2],
#                        "slope" = 1/lin$coefficients[2])
# 
g <- ggplot() + geom_point(data=cars, aes(x=mileage, y=price)) +
        geom_abline(aes(intercept = lin$coefficients[1],
                        slope = lin$coefficients[2]), color="red", size=1.25) + 
        labs(x="Mileage", y="Price [$]") +
        ggtitle("Linear Regression of Price on Mileage")
print(g)

#
# Knn (No CV) ----
#
nTrain <- round(0.9 * nrow(cars))
tInd <- sample(1:nrow(cars), nTrain)

cars.train <- cars[tInd, ]
cars.test <- cars[-tInd, ]

# estimate RMSE for all possible values of k
nearestNeighbors <- 2:100

out <- data.frame(nn = nearestNeighbors,
                  kknnRmse = rep_len(NA, length(nearestNeighbors)),
                  cvRmse = rep_len(NA, length(nearestNeighbors)))

out$kknnRmse <- sapply(out$nn, function(k) {
  cat(sprintf("%d,", k))
  kn <- kknn(price ~ mileage, train = cars.train, test = cars.test, k = k)
  rmse <- sqrt(mean((cars.test$price - kn$fitted.values)^2))
  return(rmse)
})
cat("done.\n")

print(ggplot(data=out, aes(nn, kknnRmse)) + geom_point() + geom_line() +
      labs(x="Nearest Neighbors", y="RMSE"))

print(sprintf("kknn Min RMSE=%3.2f (@%d nearest neighbors)", 
              min(out$kknnRmse), out$nn[which.min(out$kknnRmse)]))

#
# KNN (CV) ----
#
# TODO: n-fold cross validation (would be the "right" way to do this)
# code should get cleaned up, variable names follow no convetion, plots are ugly as hell

# This is the packaged way to do it, but the documentation is terrible
# "cv.kknn performs k-fold crossvalidation and is generally slower and does not
# yet contain the test of different models yet"
kn.cv <- cv.kknn(price ~ mileage, data = cars, kcv = 10)

# Home-made way to do k-fold cross validation

nfolds <- 10

folds <- cvFolds(n = nrow(cars), K = nfolds)

out$cvRmse <- sapply(out$nn, function(k) {
  cat(sprintf("%d,", k))
  
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
cat("done.\n")

# TODO: should get cleaned up and put in ggplot2

outMelted <- melt(out, id.vars = "nn")
g <- ggplot(data=outMelted, 
            aes(x=nn, y=value, group=variable, color=factor(variable))) + 
  geom_point() + geom_line() + labs(x="# of Nearest Neighbors", y="RMSE")
plot(g)
