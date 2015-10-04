# HW2

# Setup & Load Data ----
rm(list = ls())

source('../utils/source_me.R', chdir = T)
CreateDefaultPlotOpts(WriteToFile = F)

library(cvTools)
library(kknn)

dat <- read.csv(file="../data/susedcars.csv", header = T)

# 5-Fold CV ----
RunKKNNCV <- function(frm, data, nfolds, k.values) {
  fold.info <- cvFolds(nrow(data), K = nfolds)
  results.raw <- list(sqr.err = matrix(0, length(k.values), nfolds),
                      samples = matrix(0, length(k.values), nfolds))
  
  for (fold in 1:nfolds) {
    fold.test.idx <- fold.info$subset[fold.info$which == fold]
    data.test <- data[fold.test.idx, ]
    data.train <- data[-fold.test.idx, ]
    
    cat(sprintf("Fold %d => k=", fold))
    
    for (k.idx in 1:length(k.values)) {
      k <- k.values[k.idx]
      y <- data.test[,all.vars(frm)[1]]
      model <- kknn(frm, train = data.train, test = data.test, k = k)
      y.hat <- model$fitted.values
      results.raw[["sqr.err"]][k.idx, fold] <- sum((y-y.hat)^2)
      
      # track the size of each fold just in case, there's a smaller one
      results.raw[["samples"]][k.idx, fold] <- length(y)
      
      cat(sprintf("%d,", k))
    }
    cat("done\n")
  }
  
  # Aggregate some data.
  results <- list("rmse" = sqrt(rowSums(results.raw[["sqr.err"]]) / 
                                  rowSums(results.raw[["samples"]])),
                  "per.fold.rmse" = sqrt(results.raw[["sqr.err"]] / 
                                           results.raw[["samples"]]))
  return(results)
}

results.cv <- RunKKNNCV(price ~ mileage, dat, 5, seq(1, 100, 10))
