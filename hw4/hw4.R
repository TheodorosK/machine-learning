rm(list=ls())

source('../utils/source_me.R', chdir = T)
CreateDefaultPlotOpts(WriteToFile = T)

require(gamlr)
require(parallel)
require(snowfall)
require(gbm)
require(caret)
require(foreach)
require(doSNOW)
library(doMC)

sfInit(cpus=detectCores(), parallel=T)
registerDoSNOW(sfGetCluster())

LoadData <- function(what) {
  data.dir <- "../data/kdd_2009"
  prefix <- "orange_small_train"
  y <- read.delim(file.path(
    data.dir, sprintf("%s_%s.labels.txt", prefix, what)), 
    col.names="y", header = F)
  x <- read.delim(file.path(data.dir, sprintf("%s.data.gz", prefix)))
  return(cbind(y=as.factor(unlist(y)), x))
}
dat.dirty <- LoadData("churn")

# Clean Data ----
CleanData <- function(dat, na.exclude.threshold.pcnt = 1, 
                      many.factor.threshold = 50,
                      level.thresholds = data.frame(
                        name = c("low", "medium", "high"),
                        lower = c(0,     0.005, 0.010),
                        upper = c(0.005, 0.010, 0.020))) {
  # Remove Columns with all NAs
  na.count.by.var <- apply(dat, 2, function(x) sum(is.na(x)))
  all.na.vars <- which(na.count.by.var/nrow(dat) >= na.exclude.threshold.pcnt)
  print(sprintf("Removing %d columns containing >=%3.1f%% NAs: %s", 
                length(all.na.vars),
                na.exclude.threshold.pcnt * 100,
                paste(colnames(dat)[all.na.vars], collapse = ",")))
  dat <- dat[,-all.na.vars]
  
  # Apply Column Means to NAs in non-factor-columns
  for (c in which(!sapply(dat, is.factor))) {
    dat[sapply(dat[,c], is.na), c] = mean(dat[,c], na.rm = T)
  }
  if (sum(is.na(dat)) != 0) {
    stop(sprintf("Expected 0 NAs at this point, but found %d", 
                 sum(is.na(dat))))
  }
  
  # Find the columns with many levels
  for (c in which(sapply(dat, nlevels) > many.factor.threshold)) {
    before.nlevels <- nlevels(dat[,c])
    for (i in 1:nrow(level.thresholds)) {
      level.count <- table(dat[,c])     
      to.replace <-
        (level.count >= (level.thresholds[i, "lower"]*nrow(dat))) &
        (level.count <  (level.thresholds[i, "upper"]*nrow(dat)))
      levels(dat[,c])[to.replace] <- as.character(level.thresholds[i, "name"])
    }
    print(sprintf("Column %d - #levels before=%6d, after=%6d", 
                  c, before.nlevels, nlevels(dat[,c])))
  }
  
  # Col 187 is all "low"
  dat <- dat[, -c(187)]
  
  return(dat)
}
dat.clean <- CleanData(dat.dirty)

# Partition & Subsampled ----
set.seed(0x0FedBeef)
dat.partitioned <- PartitionDataset(c(0.6, 0.20, 0.20), dat.clean)
dat.subsampled <- SamplePartitionedDataset(dat.partitioned, 0.05)

# Oversample Training Data ----
OversampleData <- function(data, column) {
  partitions <- vector("list", length(data))
  for (i in 1:length(data)) {
    pdat <- data[[i]]
    
    pdat.table <- table(pdat[,column])
    rare.event.count <- sum(pdat[,column] == names(which.min(pdat.table)))
    
    dat.oversampled = NULL
    for (l in levels(pdat[,column])) {
      ldat <- pdat[pdat[,column] == l,]
      dat.oversampled <- 
        rbind(dat.oversampled, ldat[sample(1:nrow(ldat), rare.event.count),])
    }
    partitions[[i]] <- dat.oversampled
  }
  return(partitions)
}

dat.oversampled <- OversampleData(dat.partitioned, "y") 

###############################################################################
## DIMENSION REDUCTION
###############################################################################

# Importance Feature Selection ----
require(randomForest)
if (sfParallel()) {
  sfRemoveAll()
  sfExport("dat.oversampled")
}
print("Selecting Important Variables using Random Forest")
imp.rf.model <- foreach(ntree=rep(500, sfCpus()), .combine=combine,
                        .packages='randomForest') %dopar%
  randomForest(y ~ ., data=dat.oversampled[[1]], do.trace=T, importance=T)

# Should we use absolute value here?
imp.rf.model.imp <- importance(imp.rf.model, type=1)
imp.features <- rownames(imp.rf.model.imp)[
  order(imp.rf.model.imp, decreasing = T)]
imp.features <- imp.features[1:30]

PlotSetup("rf_var_sel")
varImpPlot(imp.rf.model, main = "")
PlotDone()

# Extract Important Features ----
ExtractFeatures <- function(data, features) {
  for (i in 1:length(data)) {
    p.dat <- data[[i]]
    data[[i]] <- p.dat[,colnames(p.dat) %in% features]
  }
  return(data)
} 
dat.select <- ExtractFeatures(dat.oversampled, c(imp.features, "y"))

# In honor of MT try this with a LASSO ----

library(gamlr)

GetLassoVars <- function(dat, gamma = 0, sel = "BIC") {
  if (!sel %in% c("AIC", "AICc", "BIC")) {
    stop("sel must be one of AIC, AICc, or BIC")
  }
  
  names(dat) <- c("y", sapply(names(dat)[-1], function(name) {
    paste("Var", sprintf("%03d", as.numeric(substr(name, 4, nchar(name)))), sep = "")
  }))
  
  X <- model.matrix(y ~ ., dat)[, -1]
  Y <- dat[, 1]
  levels(Y)[levels(Y) == -1] <- 0
  lin <- gamlr(X, Y, verb = T, family = "binomial", gamma = gamma)
  
  if (sel == "AIC") {
    B <- drop(coef(lin, select = which.min(AIC(lin))))[-1]
  } else if (sel == "AICc") {
    B <- drop(coef(lin, select = which.min(AICc(lin))))[-1]
  } else {
    B <- drop(coef(lin, select = which.min(BIC(lin))))[-1]    
  }
  B <- B[B != 0]
  
  sig.vars <- sapply(names(B), function(name) {
    num <- as.numeric(substr(name, 4, 6))
    paste("Var", num, sep = "")
  })
  sig.vars <- unique(sig.vars)
  
  return(unique(sig.vars))
}

lasso.vars <- GetLassoVars(dat.oversampled[[1]], gamma = 10, "BIC")
length(lasso.vars)

# Principal Components Analysis ----

GetPCAVars <- function(dat, npcs = 10) {
  X <- model.matrix(y ~ ., dat)[, -1]
  X <- X[, sapply(1:ncol(X), function(c) { var(X[, c]) != 0 })] # take out cols w/ var = 0
  
  varpc <- prcomp(X, scale = T)

  # varpc$sdev     : the standard deviations of the principal components
  # varpc$rotation : the matrix of variable loadings
  # varpc$x        : the value of the rotated data -- the Principal Components
  
  # From BD HW #7, this is how to make a scree plot (same as screeplot above)
  # Nice to go under the hood if we want to use ggplot for this
  # cc <- cov(scale(X))
  # eig <- eigen(cc)
  # barplot(sort(eig$values,decreasing=T))
  
  return(varpc$x[, 1:npcs])
}

dat.select.pca <- cbind.data.frame(as.factor(dat.oversampled[[1]][, 1]), 
                        GetPCAVars(dat.oversampled[[1]], npcs = 13))
names(dat.select.pca)[1] <- "y"

dat.validate.pca <- cbind.data.frame(as.factor(dat.partitioned[[2]][, 1]), 
                          GetPCAVars(dat.partitioned[[2]], npcs = 13))
names(dat.validate.pca)[1] <- "y"

###############################################################################
## PREDICTION
###############################################################################

# Random Forest Model ----
PredictRF <- function(dat.train, dat.validate) {
  if (sfParallel()) {
    sfRemoveAll()
    sfExport("dat.train")
  }
  rf.model <- foreach(ntree=rep(1000, sfCpus()), .combine=combine, 
                      .packages='randomForest') %dopar%
    randomForest(y ~ ., data=dat.train, ntree=ntree) 
  
  rf.model.predict <- predict(rf.model, newdata=dat.validate, type="response")
  
  conmat <- confusionMatrix(rf.model.predict, dat.validate[,"y"])
  
  return(list(rf.model, rf.model.predict, conmat))
}

dat.select.rf <- ExtractFeatures(dat.oversampled, c(imp.features, "y"))
dat.select.lasso <- ExtractFeatures(dat.oversampled, c(lasso.vars, "y"))
# PCA data frames are put together in the PCA section (they're more complicated)

print("Starting Random Forest using RF Feature Selection")
pred.rf.rf <- PredictRF(dat.select.rf[[1]], dat.partitioned[[2]])

print("Starting Random Forest using Lasso Feature Selection")
pred.rf.lasso <- PredictRF(dat.select.lasso[[1]], dat.partitioned[[2]])

print("Starting Random Forest using PCA Feature Selection")
pred.rf.pca <- PredictRF(dat.select.pca, dat.validate.pca)

# Boosting tree ----

# levels(dat.select.rf[[1]][,1])[levels(dat.select.rf[[1]][,1])=="-1"] <- "0"
# levels(dat.partitioned[[2]][,1])[levels(dat.partitioned[[2]][,1])=="-1"] <- "0"
# 
# tune.grid <-  expand.grid(interaction.depth = c(1, 3, 5),
#                         n.trees = c(500, 1000, 1500),
#                         shrinkage = c(0.01, 0.05, 0.1),
#                         n.minobsinnode = 20)
# 
# X <- dat.select.rf[[1]][,-1]
# Y <- dat.select.rf[[1]][,1]
# bst <- train(x = X, y = Y, method = "gbm", tuneGrid = tune.grid)
# 
# # Can only test on a data set that contains the same vars as the train data set
# bst.pred <- predict(bst, newdata = dat.partitioned[[2]][, names(X)])

PredictBoost <- function(dat.train, dat.validate) {

  levels(dat.train[,1])[levels(dat.train[,1])=="-1"] <- "0"
  levels(dat.validate[,1])[levels(dat.validate[,1])=="-1"] <- "0"
  
  tune.grid <-  expand.grid(interaction.depth = c(1, 5, 9),
                            n.trees = c(500, 1000, 2000),
                            shrinkage = c(0.01, 0.05, 0.1),
                            n.minobsinnode = 20)
  
  fit.control <- trainControl(## 5-fold CV
    method = "repeatedcv",
    number = 5,
    allowParallel = T)
  
  X <- dat.train[,-1]
  Y <- dat.train[,1]
  
  registerDoMC(cores = detectCores())
  bst <- train(x = X, y = Y, method = "gbm", tuneGrid = tune.grid, trControl = fit.control)
  
  # Can only test on a data set that contains the same vars as the train data set
  bst.pred <- predict(bst, newdata = dat.validate[, names(X)])

  conmat <- confusionMatrix(bst.pred, dat.validate[,"y"])  
  
  return(list(bst, bst.pred, conmat))
}

pred.boost.rf <- PredictBoost(dat.select.rf[[1]], dat.partitioned[[2]])
pred.boost.lasso <- PredictBoost(dat.select.lasso[[1]], dat.partitioned[[2]])
pred.boost.pca <- PredictBoost(dat.select.pca, dat.validate.pca)

best.tune <- rbind(c(pred.boost.rf[[1]]$bestTune[1:3], pred.boost.rf[[3]]$overall[1]),
                   c(pred.boost.lasso[[1]]$bestTune[1:3], pred.boost.lasso[[3]]$overall[1]),
                   c(pred.boost.pca[[1]]$bestTune[1:3], pred.boost.pca[[3]]$overall[1]))
best.tune <- cbind(c("Random Forest", "Gamma-Lasso", "PCA"), best.tune)

ExportTable(table=best.tune, file="boost_tune", caption="Optimal Tuning Parameters for Boosting Tree", 
            colnames=c("Dimension Reduction", "Number of Trees", "Interaction Depth", "Shrinkage", "Accuracy"), 
            display=c("s", "s", "d", "d", "f", "f"), include.rownames=F)

# Logistic regression ----

PredictGLM <- function(dat.train, dat.validate) {
  levels(dat.train[,1])[levels(dat.train[,1])=="-1"] <- "0"
  levels(dat.validate[,1])[levels(dat.validate[,1])=="-1"] <- "0"
  
  remove.cols <- c("Var207", "Var228")
  if (sum(!remove.cols %in% names(dat.train))==0) {
    dat.train <- dat.train[, !names(dat.train) %in% remove.cols]
    dat.validate <- dat.validate[, !names(dat.train) %in% remove.cols]    
  }
  
  lin <- glm(y ~ ., data = dat.train, family = "binomial")
  pred.lin <- round(predict(lin, newdata = dat.validate, type = "response"))
  
  conmat <- confusionMatrix(pred.lin, dat.validate[,"y"])  
  
  return(list(pred.lin, conmat))
}

pred.lin.rf <- PredictGLM(dat.select.rf[[1]], dat.partitioned[[2]])
pred.lin.lasso <- PredictGLM(dat.select.lasso[[1]], dat.partitioned[[2]])
pred.lin.pca <- PredictGLM(dat.select.pca, dat.validate.pca)

lin.accuracy <- c(pred.lin.rf[[2]]$overall[1],
                  pred.lin.lasso[[2]]$overall[1],
                  pred.lin.pca[[2]]$overall[1])
lin.sensitivity <- c(pred.lin.rf[[2]]$byClass[1],
                     pred.lin.lasso[[2]]$byClass[1],
                     pred.lin.pca[[2]]$byClass[1])
lin.specificity <- c(pred.lin.rf[[2]]$byClass[2],
                     pred.lin.lasso[[2]]$byClass[2],
                     pred.lin.pca[[2]]$byClass[2])
lin.results <- cbind(lin.accuracy, lin.sensitivity, lin.specificity)
rownames(lin.results) <- c("Random Forest", "Gamma-Lasso", "PCA")

ExportTable(table=lin.results, file = "glm_accuracy", 
            caption="Accuracy for Various GLM Predictions", 
            colnames=c("Accuracy", "Sensitivity", "Specificity"), 
            display=c("s", "f", "f", "f"), include.rownames=T)

# Out-of-sample predicton ----

dat.train <- rbind(dat.partitioned[[1]], dat.partitioned[[2]])
dat.test <- dat.partitioned[[3]]

dat.train <- dat.train[, c("y", lasso.vars)]
dat.test <- dat.test[, c("y", lasso.vars)]

levels(dat.train[,1])[levels(dat.train[,1])=="-1"] <- "0"
levels(dat.test[,1])[levels(dat.test[,1])=="-1"] <- "0"

tune.grid <-  expand.grid(interaction.depth = 5,
                          n.trees = 1000,
                          shrinkage = 0.01,
                          n.minobsinnode = 20)

fit.control <- trainControl(## 5-fold CV
  method = "repeatedcv",
  number = 5,
  allowParallel = T)

X <- dat.train[,-1]
Y <- dat.train[,1]

registerDoMC(cores = detectCores())
bst <- train(x = X, y = Y, method = "gbm", tuneGrid = tune.grid, trControl = fit.control)

# Can only test on a data set that contains the same vars as the train data set
bst.pred <- predict(bst, newdata = dat.test[, names(X)])

conmat <- confusionMatrix(bst.pred, dat.test[,"y"])  
