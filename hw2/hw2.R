# HW2

# Clear Vars & Load Libraries ----
rm(list = ls())

source('../utils/source_me.R', chdir = T)
CreateDefaultPlotOpts(WriteToFile = F)

library(cvTools)
library(kknn)
library(ggplot2)
library(reshape2)
library(scales)

# Load Data ----
set.seed(0x0123FEED)
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
  rmse <- data.frame(
    k = k.values,
    total = sqrt(rowSums(results.raw[["sqr.err"]]) / 
                   rowSums(results.raw[["samples"]])),
    fold = sqrt(results.raw[["sqr.err"]] / 
                  results.raw[["samples"]]))
  return(rmse)
}

k.values <- c(seq(1, 21, 3),
              seq(22, 65, 1),
              seq(65, 140, 5),
              seq(160, 400, 20))
nfolds <- 5
kknn.cv.rmse <- RunKKNNCV(price ~ mileage, dat, nfolds, k.values)
kknn.min.rmse = data.frame(
  k = kknn.cv.rmse$k[ which.min(kknn.cv.rmse$total) ], 
  rmse = kknn.cv.rmse$total[ which.min(kknn.cv.rmse$total) ])

# Plot CV results ----
plot.colors <- gg_color_hue(nfolds+1)
kknn.cv.melt <- melt(kknn.cv.rmse, id.vars='k')

g <- ggplot() +
  # Plots RMSEs for each K
  geom_point(data=kknn.cv.melt, aes(k, value, color=variable)) +
  geom_line(data=kknn.cv.melt, aes(k, value, color=variable), size=1.25) +
  scale_color_manual(
    name="Run",
    values=c(plot.colors[1], alpha(plot.colors[2:(nfolds+1)], 0.15)),
    labels = c("Combined", sprintf("Fold #%d", 1:nfolds))) +
  # Show the mins
  geom_point(data = kknn.min.rmse, aes(x=k, y=rmse), 
             pch=21, size=4, color=plot.colors[1]) +
  geom_text(data = kknn.min.rmse, hjust=0, vjust=0,
            aes(x=k, y=rmse, label=sprintf("Minimum @ k=%d", kknn.min.rmse$k))) +
  # Misc cleanup/labels
  labs(x="k", y="RMSE") + theme_bw() + theme(legend.position = c(0.8, 0.75)) +
  scale_x_continuous(expand=c(0.01, 0))

PlotSetup("knn_cv_min_k")
plot(g)
PlotDone()