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

set.seed(926)
dat <- read.csv(file="../data/susedcars.csv", header = T)

# N-Fold CV ----
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
      model <- kknn(frm, train = data.train, test = data.test, k = k, 
                    kernel = 'rectangular')
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
    
    # This is what is in the lecture notes
    total = rowMeans(sqrt(results.raw[["sqr.err"]] / 
                            results.raw[["samples"]])),
    
    # This is what I would do
    #     total = sqrt(rowSums(results.raw[["sqr.err"]]) / 
    #                     rowSums(results.raw[["samples"]])),
    
    # Per-fold RMSE
    fold = sqrt(results.raw[["sqr.err"]] / 
                  results.raw[["samples"]]))
  return(rmse)
}

k.values <- c(seq(1, 13, 3),
              seq(14, 50, 1),
              seq(55, 140, 5),
              seq(160, 400, 20))
nfolds <- 5
kknn.cv.rmse <- RunKKNNCV(price ~ mileage, dat, nfolds, k.values)
kknn.cv.min.rmse = data.frame(
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
    name = "Run",
    values=c(plot.colors[1], alpha(plot.colors[2:(nfolds+1)], 0.15)),
    labels = c("Combined", sprintf("Fold #%d", 1:nfolds))) +
  # Show the mins
  geom_point(data = kknn.cv.min.rmse, aes(x=k, y=rmse), 
             pch=21, size=4, color=plot.colors[1]) +
  geom_text(data = kknn.cv.min.rmse, hjust=0, vjust=0,
            aes(x=k, y=rmse, label=sprintf("Minimum @ k=%d", kknn.cv.min.rmse$k))) +
  # Misc cleanup/labels
  labs(x="k", y="RMSE") + theme_bw() + theme(legend.position = c(0.8, 0.75)) +
  scale_x_continuous(expand=c(0.01, 0))

PlotSetup("min_k_cv")
plot(g)
PlotDone()

# Plot k's ----
fittedK <- function(frm, k, data) {
  test = data[order(data[,all.vars(frm)[2]]),]
  kn <- kknn(frm, train = data, test = test, k = k)
  return(data.frame(mileage = test$mileage, 
                    price_hat = kn$fitted.values))
}

kknn.eye.k <- 40
kknn.eye.fit <- fittedK(price ~ mileage, kknn.eye.k, dat)
kknn.cv.fit <- fittedK(price ~ mileage, kknn.cv.min.rmse$k, dat)

kknn.price_hat.melt <- melt(data.frame(mileage = kknn.eye.fit$mileage, 
                                       eye = kknn.eye.fit$price_hat,
                                       cv = kknn.cv.fit$price_hat), 
                            id.vars="mileage", value.name = "price")

g <- ggplot() + geom_point(data=dat, aes(x=mileage/1000, y=price/1000), alpha=0.20) +
  geom_line(data = kknn.price_hat.melt, size=1, alpha=0.8, 
            aes(x=mileage/1000, y=price/1000, color=variable)) +
  scale_color_discrete(name="Method", 
                       labels = c(sprintf("Eyeball (k=%d)", kknn.eye.k),
                                  sprintf("CV (k=%d)", kknn.cv.min.rmse$k))) +
  theme_bw() + theme(legend.position = c(0.8, 0.75)) +
  labs(x="Mileage [1000 miles]", y="Price [1000 $]") +
  scale_x_continuous(expand=c(0,0)) + scale_y_continuous(expand=c(0,0))
PlotSetup("fit_eye_cv")
plot(g)
PlotDone()
