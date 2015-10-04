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

k.values <- c(seq(1, 60, 1),
              seq(65, 100, 5),
              seq(120, 400, 20))
nfolds <- 5
rmse.cv <- RunKKNNCV(price ~ mileage, dat, nfolds, k.values)

# Plot CV results ----
plot.colors <- gg_color_hue(nfolds+1)
min.dat = data.frame(k = rmse.cv$k[which.min(rmse.cv$total)], 
                     rmse = rmse.cv$total[which.min(rmse.cv$total)])

g <- ggplot() +
  geom_line(data=melt(rmse.cv, id.vars='k'), aes(k, value, color=variable), size=1) +
  scale_color_manual(name="Run",
                     values=c(plot.colors[1], alpha(plot.colors[2:(nfolds+1)], 0.3)),
                     labels = c("Total", sprintf("Fold #%d", 1:nfolds))) +
  geom_point(data = min.dat, aes(x=k, y=rmse), pch=21, size=4, color=plot.colors[1]) +
  geom_text(data = min.dat, hjust=0, vjust=0,
            aes(x=k, y=rmse, label=sprintf("Minimum @ k=%d", min.dat$k))) +
  labs(x="k", y="RMSE") + theme_bw()
plot(g)
