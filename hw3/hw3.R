# HW3

# Clear Vars & Load Libraries ----
rm(list = ls())

library(rpart)
library(randomForest)
library(gbm)
library(ggplot2)

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
  
  tree <- rpart(form = form, data = data, 
                control = rpart.control(minsplit = 5, cp = 0.0005))
  
  idx.best = which.min(tree$cptable[, "xerror"])
  cp.best <- tree$cptable[idx.best, "CP"]
  
  tree.prune <- prune(tree, cp = cp.best)
  
  return(list(tree, tree.prune))
  
}

trees.small <- FitAndPruneTree(price ~ mileage, cars.train)
tree.small <- trees.small[1][[1]]
tree.small.prune <- trees.small[2][[1]]

trees.big <- FitAndPruneTree(price ~ ., cars.train)
tree.big <- trees.big[1][[1]]
tree.big.prune <- trees.big[2][[1]]

# Visualize
require(rpart.plot)

PlotSetup("tree_small")
rpart.plot(tree.small)
PlotDone()

PlotSetup("tree_small_prune")
rpart.plot(tree.small.prune)
PlotDone()

# Bagging MP ---------------------------------------------------------------------
# run an experiment to see if predictive accuracy increases with B
cat("\n\n-----bootstrap [MP]-----\n")

RunBootStrap <- function(B, parallel=T) {
  require(snowfall)
  
  sfInit(cpus = detectCores(), parallel = parallel)
  if (sfParallel()) { 
    sfRemoveAll()
    sfExport(list=c("cars.train", "cars.val"))
  }
  sfLibrary(rpart)
  par.results <- sfClusterApplyLB(1:B, function(b) {
    cat(sprintf("%d,", b))
    # bootstrap sample with replacement
    bsamp <- sample(1:nrow(cars.train), nrow(cars.train), replace = T)

    # do not prune trees!
    tree.small = rpart(price ~ mileage, data = cars.train[bsamp, ])
    tree.large = rpart(price ~ ., data = cars.train[bsamp, ])
    
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
#   bsamp <- sample(1:nrow(cars.train), nrow(cars.train), replace = T) # bootstrap sample with replacement
#   
#   # do not prune trees!
#   tree.small = rpart(price ~ mileage, data = cars.train[bsamp, ])
#   tree.big = rpart(price ~ ., data = cars.train[bsamp, ])
#   
#   bs.pred.small[, b] <- predict(tree.small, newdata = cars.val)
#   bs.pred.big[, b] <- predict(tree.big, newdata = cars.val)
#   
# }
# cat("done.\n")

# Random forest ---------------------------------------------------------------
# Should come back and make a funciton out of this

cat("\n\n-----random forest-----\n\n")

rf.small <- randomForest(price ~ mileage, data = cars.train, do.trace = 20)

# rf.big <- randomForest(price ~ ., data = cars.train, do.trace = 20, ntree = 250)

ncores <- detectCores()
ntree.par <- ceiling(500/8)

sfInit(cpus = ncores, parallel = T)
if (sfParallel()) { 
  sfRemoveAll()
  sfExport(list=c("cars.train", "ntree.par"))
}
sfLibrary(randomForest)
rf.par <- sfClusterApplyLB(1:ncores, function(t) {
  rf <- randomForest(price ~ ., data = cars.train, do.trace = 20, ntree = ntree.par)
  return(rf)
})
sfStop()

rf.big <- combine(rf.par[[1]], rf.par[[2]], rf.par[[3]], rf.par[[4]], 
                   rf.par[[5]], rf.par[[6]], rf.par[[7]], rf.par[[8]])

# Visualize out-of-bag RMSE (can't do this on a combined tree)
rf <- randomForest(price ~ ., data = cars.train, do.trace = 20, ntree = 500)

rf.pdat <- data.frame(tree = c(1:length(rf$mse)), mse = rf$mse)
g <- ggplot(rf.pdat, aes(tree, mse)) + geom_line(size = 2) + 
  labs(x = "Number of Trees", y = "Out-of-Bag Mean Square Error") + 
  theme_bw()
PlotSetup("rf_oob_mse")
print(g)
PlotDone()

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
                     data = cars.train, n.trees = test.params[i], 
                     interaction.depth = boost.indepth,
                     shrinkage = boost.shrink) 
      } else if (test.param == "interaction.depth") {
        boost <- gbm(formula = price ~ ., distribution = "gaussian", 
                     data = cars.train, n.trees = boost.ntree, 
                     interaction.depth = test.params[i],
                     shrinkage = boost.shrink) 
      } else if (test.param == "shrinkage") {
        boost <- gbm(formula = price ~ ., distribution = "gaussian", 
                     data = cars.train, n.trees = boost.ntree, 
                     interaction.depth = boost.indepth,
                     shrinkage = test.params[i]) 
      } else {
        stop("That is not a valid test parameter")
      } 
      
      price.hat <- predict(boost, newdata = cars.val, n.trees = boost.ntree)
      rmse[i] <- mean(sqrt((cars.val$price - price.hat)^2))
      
    } # end for loop
    
    return(rmse)
    
  })
  sfStop()

  return(data.frame(
    test.params = test.params,
    rmse.trial = do.call(cbind, boost.test)))
}

PlotGBMParamSweep <- function(data, param.name) {
  rmse.means <- data.frame(param = data[,1], 
                           rmse = rowMeans(data[,-1]))
  rmse.sd <- apply(data[,-1], 1, sd)
  rmse.min <- rmse.means$param[which.min(rmse.means$rmse)]
  rmse.1se <- rmse.means$param[min(which(rmse.means$rmse < (
    min(rmse.means$rmse) + rmse.sd[which.min(rmse.means$rmse)])))]
  sel <- data.frame(param = c("min", "1se"),
                    value = c(rmse.min, rmse.1se))
  sel$param <- sprintf("%s @%.2f", sel$param, sel$value)
  require(ggplot2)
  require(reshape2)
  g <- ggplot(data=rmse.means, aes(x=param, y=rmse)) + geom_point(size=3) + geom_line(lwd=1.5) +
    geom_vline(data=melt(sel, id.vars="param"), aes(xintercept=value, linetype=param), show_guide = T) +
    labs(x=param.name, y="RMSE") + theme_bw() + 
    guides(linetype = guide_legend(title = "Selection\nCriteria")) +
    theme(legend.position = c(0.85, 0.8))
  plot(g)
}


boost.ntree = 50
boost.indepth = 16
boost.shrink = 0.11

seq.ntree <- seq(20, 150, 10)
seq.indepth <- seq(2, 28)
seq.shrink <- seq(0.01, 0.5, 0.02)

export.list = c("cars.train", "cars.val", "seq.ntree", "seq.indepth", "seq.shrink",
                "boost.ntree", "boost.indepth", "boost.shrink")

# What is the optimal n.trees?
# This is so noisy!!!
test.ntree <- LoadCachedOrRun(TestGBM, export.list, "n.trees", 
                              seq.ntree, nTrials = 100)
PlotSetup("gbm_ntree")
PlotGBMParamSweep(test.ntree, "# of Trees")
PlotDone()

# cat(sprintf("the \"optimal\" value of n.trees is %.1f", mean(test.ntree)))
# PlotSetup("histo_ntree")
# hist(test.ntree)
# PlotDone()

# What is the optimal interaciton.depth?
test.indepth <- LoadCachedOrRun(TestGBM, export.list, "interaction.depth", 
                                seq.indepth, nTrials = 100)
PlotSetup("gbm_indepth")
PlotGBMParamSweep(test.indepth, "Interaction Depth")
PlotDone()

# cat(sprintf("the \"optimal\" value of interaction.depth is %.1f", mean(test.indepth)))
# PlotSetup("histo_indepth")
# hist(test.indepth)
# PlotDone()

# What is the optimal shrinkage parameter?
test.shrink <- LoadCachedOrRun(TestGBM, export.list, "shrinkage", 
                               seq.shrink, nTrials = 100)
PlotSetup("gbm_shrink")
PlotGBMParamSweep(test.shrink, "Crush Factor")
PlotDone()

# cat(sprintf("the \"optimal\" value of shrinkage param is %.1f", mean(test.shrink)))
# PlotSetup("histo_shrink")
# hist(test.shrink)
# PlotDone()

# Now with the "tuned" parameters

boost.small <- gbm(formula = price ~ mileage, distribution = "gaussian", 
                   data = cars.train, n.trees = boost.ntree, 
                   interaction.depth = boost.indepth,
                   shrinkage = boost.shrink, verbose = T)

# 1.478 secs
boost.big <- gbm(formula = price ~ ., distribution = "gaussian", 
                 data = cars.train, n.trees = boost.ntree, 
                 interaction.depth = boost.indepth,
                 shrinkage = boost.shrink, verbose = T)

# Multiple Regression ----
require(gamlr)

doGamlr <- function(frm, data.train, data.valid, plot.name, ...) {
  resp <- model.response(model.frame(frm, data=data.train))
  mm <- model.matrix(frm, data.train)[,-1]
  lin.model <- cv.gamlr(mm, resp, verb = T, ...)
  
  PlotSetup(plot.name)
  plot(lin.model)
  PlotDone()

  x <- model.matrix(frm, data.valid)[,-1]
  return(predict(lin.model, x, select="1se"))
}
lin.reg <- drop(exp(doGamlr(
  log(price) ~ . + .^2, 
  cars.train, cars.val, "lin_reg_interaction", 
  lambda.min=exp(-9))))
lin.reg_simple <- unlist(exp(doGamlr(
  log(price) ~ ., 
  cars.train, cars.val, "lin_reg_simple",
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
cat("best model:", names(predict.val)[which.min(rmse.val)], "\n")

rmse.val <- rmse.val[order(rmse.val)]
print(rmse.val)
model.names <- c("Boosting Tree (big)", "Random Forest (big)", "LASSO Regression (big)", 
                 "Bagging (big)", "Regression Tree (big)", "Pruned Regression Tree (big)", 
                 "Boosting Tree (small)", "Bagging (small)", "Regression Tree (small)", 
                 "Pruned Regression Tree (small)", "Random Forest (small)")
df.rmse <- data.frame(models = model.names, rmse = rmse.val)

ExportTable(table = df.rmse, file = "rmse_comp", 
            caption = "Comparison of RMSE for Various Models", 
            colnames = c("Model", "RMSE"), include.rownames = F)

# Try the 'best' model on the test set

df.rmse.test <- data.frame(models = c("Boosting Tree", "Random Forest", 
                                      "LASSO Regression"), 
                           rmse.val = rmse.val[1:3], rmse.test = rep(NA, 3))
rownames(df.rmse.test) <- NULL

# Boosting
boost.best <- gbm(formula = price ~ ., distribution = "gaussian", 
                  data = rbind(cars.train, cars.val), 
                  n.trees = boost.ntree, 
                  interaction.depth = boost.indepth,
                  shrinkage = boost.shrink, verbose = T)
boost.yhat <- predict(boost.best, newdata = cars.test, n.trees = boost.ntree)
df.rmse.test$rmse.test[1] <- mean(sqrt((cars.test$price - boost.yhat)^2))

# Random forest
rf.best <- randomForest(price ~ ., data = rbind(cars.train, cars.val), 
                        do.trace = 20, ntree = 250)
rf.yhat <- predict(rf.best, newdata = cars.test)
df.rmse.test$rmse.test[2] <- mean(sqrt((cars.test$price - rf.yhat)^2))

# LASSO
lin.yhat <- unlist(exp(doGamlr(
  log(price) ~ ., 
  rbind(cars.train, cars.val), cars.test, "lin_reg_best",
  lambda.min=exp(-9))))
df.rmse.test$rmse.test[3] <- mean(sqrt((cars.test$price - lin.yhat)^2))

ExportTable(table = df.rmse.test, file = "rmse_test", 
            caption = "OOS RMSE for Top Three Models", 
            colnames = c("Model", "RMSE (Validate)", "RMSE (Test)"), 
            include.rownames = F)
