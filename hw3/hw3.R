# HW3

# Clear Vars & Load Libraries ----
rm(list = ls())

library(rpart)
library(randomForest)
library(gbm)

source('../utils/source_me.R', chdir = T)
CreateDefaultPlotOpts(WriteToFile = T)

set.seed(1010)

# Cars data set: goal is to predict price
# Split into train (0.5), validate (0.25), test (0.25)
cars <- read.csv(file = "../data/usedcars.csv")
cars.partitioned <- PartitionDataset(c(0.5, 0.25, 0.25), cars)

cars.train <- cars.partitioned[[1]]
cars.val <- cars.partitioned[[2]]
cars.test <- cars.partitioned[[3]]

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

trees.small <- FitAndPruneTree(price ~ mileage, cars.test)
tree.small <- trees.small[1][[1]]
tree.small.prune <- trees.small[2][[1]]

trees.big <- FitAndPruneTree(price ~ ., cars.test)
tree.big <- trees.big[1][[1]]
tree.big.prune <- trees.big[2][[1]]

# Bagging MP ---------------------------------------------------------------------
# run an experiment to see if predictive accuracy increases with B
cat("\n\n-----bootstrap [MP]-----\n")

RunBootStrap <- function(B, parallel=T) {
  require(snowfall)
  
  sfInit(cpus = detectCores(), parallel = parallel)
  if (sfParallel()) { 
    sfRemoveAll()
    sfExport(list=c("cars.test", "cars.val"))
  }
  sfLibrary(rpart)
  par.results <- sfClusterApplyLB(1:B, function(b) {
    cat(sprintf("%d,", b))
    # bootstrap sample with replacement
    bsamp <- sample(1:nrow(cars.test), nrow(cars.test), replace = T)

    # do not prune trees!
    tree.small = rpart(price ~ mileage, data = cars.test[bsamp, ])
    tree.large = rpart(price ~ ., data = cars.test[bsamp, ])
    
    return(list(small=predict(tree.small, newdata = cars.val),
                large=predict(tree.large, newdata = cars.val)))
  })
  cat("done\n")
  sfStop()
  
  # Combine parallel results.
  pred.small <- matrix(data = NA, nrow = nrow(cars.val), ncol = B)
  pred.large <- matrix(data = NA, nrow = nrow(cars.val), ncol = B)
  for (b in 1:B) {
    pred.small[,b] <- par.results[[b]][["small"]]
    pred.large[,b] <- par.results[[b]][["large"]]
  }
  return(list(small=pred.small,
              large=pred.large))
}
bs.results <- RunBootStrap(100)
bs.pred.small <- bs.results[["small"]]
bs.pred.big <- bs.results[["large"]]

# # Bagging ----
# 
# cat(sprintf("B = %d\n\n", B))
# for (b in 1:B) {
#   
#   cat(sprintf("%d,", b))
#   
#   bsamp <- sample(1:nrow(cars.test), nrow(cars.test), replace = T) # bootstrap sample with replacement
#   
#   # do not prune trees!
#   tree.small = rpart(price ~ mileage, data = cars.test[bsamp, ])
#   tree.big = rpart(price ~ ., data = cars.test[bsamp, ])
#   
#   bs.pred.small[, b] <- predict(tree.small, newdata = cars.val)
#   bs.pred.big[, b] <- predict(tree.big, newdata = cars.val)
#   
# }
# cat("done.\n")

# Random forest ---------------------------------------------------------------
# Look carefully at mtry; 3 gives me a warning but the default runs fine:
# for price ~ mileage => p=1.  mtry the number of randomly sampled covariates 
# test at each branch must be between [1, p];
# Can definitely try a higher number of nodes for the price ~ .
#
# We could also do some experiments with ntree.

cat("\n\n-----random forest-----\n\n")
rf.small <- randomForest(price ~ mileage, data = cars.test, do.trace = 20)
rf.big <- randomForest(price ~ ., data = cars.test, do.trace = 20)

# Boosting --------------------------------------------------------------------
# play with interaction.depth, n.trees, shrinkage
# some notes:
# -- n.trees optimal value (for big tree) is around 41; for boost.small there 
#    is lots of run-to-run variation however, not sure what the cause is (don't 
#    understand boosting that well I guess)
# -- interaction.depth optimal value around 9, w/ some r2r variation again
# -- shrinkage optimal around 0.2, but lots of r2r variation
# -- so far I'm assuming that there is no interaction effect of tuning params
#    to really test this, need to simulate in 3d space, right?

TestGBM <- function(export.list, test.param, test.range) {
  sfInit(cpus = detectCores(), parallel = T)
  if (sfParallel()) { 
    sfRemoveAll()
    sfExport(list=export.list)
  }
  sfLibrary(gbm)
  rmse.boost <- sfClusterApplyLB(1:length(test.range), function(i) {
    if (test.param == "n.trees") {
      boost <- gbm(formula = price ~ ., distribution = "gaussian", 
                   data = cars.test, n.trees = test.range[i], 
                   interaction.depth = boost.indepth,
                   shrinkage = boost.shrink) 
    } else if (test.param == "interaction.depth") {
      boost <- gbm(formula = price ~ ., distribution = "gaussian", 
                   data = cars.test, n.trees = boost.ntree, 
                   interaction.depth = test.range[i],
                   shrinkage = boost.shrink) 
    } else if (test.param == "shrinkage") {
      boost <- gbm(formula = price ~ ., distribution = "gaussian", 
                   data = cars.test, n.trees = boost.ntree, 
                   interaction.depth = boost.indepth,
                   shrinkage = test.range[i]) 
    } else {
      stop("That is not a valid test parameter")
    } 
    price.hat <- predict(boost, newdata = cars.val, n.trees = boost.ntree)
    return(mean(sqrt((cars.val$price - price.hat)^2)))
  })
  cat("done\n")
  sfStop()
  return(unlist(rmse.boost))    
}

boost.ntree = 70
boost.indepth = 9
boost.shrink = 0.16

test.ntree <- seq(30, 100, 10)
test.indepth <- seq(5, 15)
test.shrink <- seq(0.01, 0.5, 0.02)
export.list = c("cars.test", "cars.val", "test.ntree", "test.indepth", "test.shrink",
                "boost.ntree", "boost.indepth", "boost.shrink")

rmse.boost <- TestGBM(export.list, "n.trees", test.ntree)
cat(sprintf("minimum rmse when using n.tree = %d", 
            test.ntree[which.min(rmse.boost)]))

# Try to smooth out the run-to-run variation (this takes a while)
# Q: is it wrong to parallelize one layer below this? 
T <- 100
opt.ntree <- rep(NA, 10)
for (i in 1:T) {
  cat(sprintf("\n\n- - - - %d - - - -\n\n", i))
  rmse.boost <- TestGBM(export.list, "n.trees", test.ntree)
  opt.ntree[i] <- test.ntree[which.min(rmse.boost)]
}
cat("done\n")

# Now with the "tuned" parameters

boost.small <- gbm(formula = price ~ mileage, distribution = "gaussian", 
                   data = cars.test, n.trees = boost.ntree, 
                   interaction.depth = boost.indepth,
                   shrinkage = boost.shrink, verbose = T)

boost.big <- gbm(formula = price ~ ., distribution = "gaussian", 
                 data = cars.test, n.trees = boost.ntree, 
                 interaction.depth = boost.indepth,
                 shrinkage = boost.shrink, verbose = T)

# Multiple Regression ----
require(gamlr)

doGamlr <- function(frm, data.train, data.valid, ...) {
  resp <- model.response(model.frame(frm, data=data.train))
  mm <- model.matrix(frm, data.train)[,-1]
  lin.model <- cv.gamlr(mm, resp, verb = T, ...)
  plot(lin.model)

  x <- model.matrix(frm, data.valid)[,-1]
  return(predict(lin.model, x, select="1se"))
}
lin.reg <- drop(exp(doGamlr(
  log(price) ~ . + .^2, 
  cars.train, cars.val, 
  lambda.min=exp(-9))))

# Predict ---------------------------------------------------------------------

predict.val = data.frame(
  tree.small = predict(tree.small, newdata = cars.val),
  tree.small.prune = predict(tree.small.prune, newdata = cars.val),
  tree.big = predict(tree.big, newdata = cars.val),
  tree.big.prune = predict(tree.big.prune, newdata = cars.val),
  bag.small = rowMeans(bs.pred.small),
  bag.big = rowMeans(bs.pred.big),
  rf.small = predict(rf.small, newdata = cars.val),
  rf.big = predict(rf.big, newdata = cars.val),
  lin.reg = lin.reg, 
  boost.small = predict(boost.small, newdata = cars.val, n.trees = boost.ntree),
  boost.big = predict(boost.big, newdata = cars.val, n.trees = boost.ntree)
)

rmse.val <- sapply(1:ncol(predict.val), function(c) {
  mean(sqrt((cars.val$price - predict.val[, c])^2))
})
names(rmse.val) <- names(predict.val)

cat("best model:", names(predict.val)[which.min(rmse.val)])

print(rmse.val[order(rmse.val)])
