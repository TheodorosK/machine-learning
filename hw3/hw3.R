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

TestGBM <- function(export.list, test.param, test.params, nTrials) {
  
  sfInit(cpus = detectCores(), parallel = T)
  if (sfParallel()) { 
    sfRemoveAll()
    sfExport(list=export.list)
  }
  sfLibrary(gbm)
  boost.test <- sfClusterApplyLB(1:nTrials, function(t) {
    
    rmse <- rep(NA, length(test.params))
    for (i in 1:length(test.params)) {
      if (test.param == "n.trees") {
        boost <- gbm(formula = price ~ ., distribution = "gaussian", 
                     data = cars.test, n.trees = test.params[i], 
                     interaction.depth = boost.indepth,
                     shrinkage = boost.shrink) 
      } else if (test.param == "interaction.depth") {
        boost <- gbm(formula = price ~ ., distribution = "gaussian", 
                     data = cars.test, n.trees = boost.ntree, 
                     interaction.depth = test.params[i],
                     shrinkage = boost.shrink) 
      } else if (test.param == "shrinkage") {
        boost <- gbm(formula = price ~ ., distribution = "gaussian", 
                     data = cars.test, n.trees = boost.ntree, 
                     interaction.depth = boost.indepth,
                     shrinkage = test.params[i]) 
      } else {
        stop("That is not a valid test parameter")
      } 
      
      price.hat <- predict(boost, newdata = cars.val, n.trees = boost.ntree)
      rmse[i] <- mean(sqrt((cars.val$price - price.hat)^2))
      
    } # end for loop
    
    return(test.params[which.min(rmse)])
    
  })
  sfStop()
  
  return(unlist(boost.test))
  
}

boost.ntree = 80
boost.indepth = 9
boost.shrink = 0.16

export.list = c("cars.test", "cars.val", "seq.ntree", "seq.indepth", "seq.shrink",
                "boost.ntree", "boost.indepth", "boost.shrink")

# What is the optimal n.trees?
# This is so noisy!!!
seq.ntree <- seq(20, 150, 10)
test.ntree <- TestGBM(export.list, "n.trees", seq.ntree, nTrials = 1000)
cat(sprintf("the \"optimal\" value of n.trees is %.1f", mean(test.ntree)))
PlotSetup("histo_ntree")
hist(test.ntree)
PlotDone()

# What is the optimal interaciton.depth?
seq.indepth <- seq(5, 15)
test.indepth <- TestGBM(export.list, "interaction.depth", seq.indepth, nTrials = 1000)
cat(sprintf("the \"optimal\" value of interaction.depth is %.1f", mean(test.indepth)))
PlotSetup("histo_indepth")
hist(test.indepth)
PlotDone()

# What is the optimal shrinkage parameter?
seq.shrink <- seq(0.01, 0.5, 0.02)
test.shrink <- TestGBM(export.list, "shrinkage", seq.shrink, nTrials = 1000)
cat(sprintf("the \"optimal\" value of shrinkage param is %.1f", mean(test.shrink)))
PlotSetup("histo_shrink")
hist(test.shrink)
PlotDone()

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
