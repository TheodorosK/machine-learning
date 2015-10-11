# HW3

# Clear Vars & Load Libraries ----
rm(list = ls())

library(rpart)
library(randomForest)
library(gbm)
library(gamlr)

source('../utils/source_me.R', chdir = T)
CreateDefaultPlotOpts(WriteToFile = T)

set.seed(1010)

# Cars data set: goal is to predict price
# Split into train (0.5), validate (0.25), test (0.25)
cars <- read.csv(file = "../data/usedcars.csv")

n0 = nrow(cars)
n1 = floor(n0/2)
n2 = floor(n0/4)
n3 = n0 - n1 - n2

idx <- sample(1:n0, n0)

cars.train <- cars[idx[1:n1], ]
cars.val <- cars[idx[n1+1:n2], ]
cars.test <- cars[idx[n1+n2+1:n3], ]

# Trees -----------------------------------------------------------------------

# rpart returns rpart.object -- see ?rpart.object for more
# slides use other options, need to investigate

FitAndPruneTree <- function(form, data) {
  
  tree <- rpart(form = form, data = data)

  idx.best = which.min(tree$cptable[, "xerror"])
  cp.best <- tree$cptable[idx.best, "CP"]
  
  tree.prune <- prune(tree, cp = cp.best)
  
  return(list(tree, tree.prune))

}

trees.small <- FitAndPruneTree("price ~ mileage", cars.test)
tree.small <- trees.small[1][[1]]
tree.small.prune <- trees.small[2][[1]]

trees.big <- FitAndPruneTree("price ~ .", cars.test)
tree.big <- trees.big[1][[1]]
tree.big.prune <- trees.big[2][[1]]

# Predict ---------------------------------------------------------------------

predict.val = data.frame(
  price.obs = cars.val$price,
  tree.small = predict(tree.small, newdata = cars.val),
  tree.small.prune = predict(tree.small.prune, newdata = cars.val),
  tree.big = predict(tree.big, newdata = cars.val),
  tree.big.prune = predict(tree.big.prune, newdata = cars.val)
)
