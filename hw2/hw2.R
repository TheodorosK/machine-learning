# HW2

# Clear Vars & Load Libraries ----
rm(list = ls())

source('../utils/source_me.R', chdir = T)
CreateDefaultPlotOpts(WriteToFile = T)

library(cvTools)
library(kknn)
library(ggplot2)
library(reshape2)
library(scales)
library(parallel)
library(tictoc)

set.seed(926)
dat <- read.csv(file="../data/susedcars.csv", header = T)

# Set CV parameters
nfolds <- 5
k.values <- c(seq(1, 13, 3),
              seq(14, 50, 1),
              seq(55, 140, 5),
              seq(160, 300, 20))

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

cv.1p.rmse <- RunKKNNCV(price ~ mileage, dat, nfolds, k.values)
cv.1p.min.rmse = data.frame(
  k = cv.1p.rmse$k[ which.min(cv.1p.rmse$total) ], 
  rmse = cv.1p.rmse$total[ which.min(cv.1p.rmse$total) ])

# See how much run-to-run variation there is ----

# From the parallel docs:
# https://stat.ethz.ch/R-manual/R-devel/library/parallel/doc/parallel.pdf
# The alternative is to set separate seeds for each worker process in some 
# reproducible way from the seed in the master process. This is generally 
# plenty safe enough, but there have been worries that the random-number 
# streams in the workers might somehow get into step.

VarianceKKNN <- function(T, nfolds) {
  ncores <- detectCores()
  clust <- makeCluster(ncores, type = "FORK", outfile = "status.txt")
  seeds <- sample(1:1000, size = T)
  min.k <- parSapply(cl = clust, X = 1:T, FUN = function(t) {
    cat(sprintf("\n\n\n * * * * * * * * %d * * * * * * * * \n\n\n", t))
    set.seed(seeds[t])
    cv.1p.rmse <- RunKKNNCV(price ~ mileage, dat, nfolds, k.values)
    return(cv.1p.rmse$k[which.min(cv.1p.rmse$total)])
  })
  stopCluster(clust)  
  return(min.k)
}

kseq.5 <- VarianceKKNN(T = 100, nfolds = 5)
kseq.10 <- VarianceKKNN(T = 100, nfolds = 10)
kseq.20 <- VarianceKKNN(T = 100, nfolds = 20)

kseq <- cbind(kseq.5, kseq.10, kseq.20)
names(kseq) <- c("nfold5", "nfold10", "nfold20")

# Plot variation of k's
kseq.melt <- melt(data = kseq, value.name = "k")
names(kseq.melt)[1:2] <- c("Index", "nFolds")

# g <- ggplot(kseq.melt, aes(x = Index, y = k, color = nFolds)) + geom_point()
# print(g)

g <- ggplot(kseq.melt, aes(nFolds, k)) + geom_boxplot()
print(g)

# Plot CV results ----
plot_cv_rmse <- function(rmse, min.rmse, nfolds) {
  plot.colors <- gg_color_hue(nfolds+1)
  rmse.melt <- melt(rmse, id.vars='k')
  
  min.rmse$label <- sprintf("Minimum @ k=%d", min.rmse$k)
  
  g <- ggplot() +
    # Plots RMSEs for each K
    geom_point(data=rmse.melt, aes(k, value, color=variable)) +
    geom_line(data=rmse.melt, aes(k, value, color=variable), size=1.25) +
    scale_color_manual(
      name = "Run",
      values=c(plot.colors[1], alpha(plot.colors[2:(nfolds+1)], 0.15)),
      labels = c("Combined", sprintf("Fold #%d", 1:nfolds))) +
    # Show the mins
    geom_point(data = min.rmse, aes(x=k, y=rmse), 
               pch=21, size=4, color=plot.colors[1]) +
    geom_text(data = min.rmse, hjust=0, vjust=0,
              aes(x=k, y=rmse, label=label)) +
    # Misc cleanup/labels
    labs(x="k", y="RMSE")
  return(g)
}

PlotSetup("1p_cv_k")
g <- plot_cv_rmse(cv.1p.rmse, cv.1p.min.rmse, nfolds) + 
  theme_bw() + theme(legend.position=c(0.8, 0.75)) +
  scale_x_continuous(expand=c(0.01, 0))
plot(g)
PlotDone()

# Plot k's ----
fittedK <- function(frm, k, data) {
  test = data[order(data[,all.vars(frm)[2]]),]
  kn <- kknn(frm, train = data, test = test, k = k)
  return(data.frame(mileage = test$mileage, 
                    price_hat = kn$fitted.values))
}

eye.1p.k <- 40
eye.1p.fit <- fittedK(price ~ mileage, eye.1p.k, dat)
cv.1p.fit <- fittedK(price ~ mileage, cv.1p.min.rmse$k, dat)

kknn.price_hat.melt <- melt(data.frame(mileage = eye.1p.fit$mileage, 
                                       eye = eye.1p.fit$price_hat,
                                       cv = cv.1p.fit$price_hat), 
                            id.vars="mileage", value.name = "price")

g <- ggplot() + geom_point(data=dat, aes(x=mileage/1000, y=price/1000), alpha=0.20) +
  geom_line(data = kknn.price_hat.melt, size=1, alpha=0.8, 
            aes(x=mileage/1000, y=price/1000, color=variable)) +
  scale_color_discrete(name="Method", 
                       labels = c(sprintf("Eyeball (k=%d)", eye.1p.k),
                                  sprintf("CV (k=%d)", cv.1p.min.rmse$k))) +
  theme_bw() + theme(legend.position = c(0.8, 0.75)) +
  labs(x="Mileage [1000 miles]", y="Price [1000 $]") +
  scale_x_continuous(expand=c(0,0)) + scale_y_continuous(expand=c(0,0))
PlotSetup("1p_fit_eye_ev")
plot(g)
PlotDone()

# Scale Data ----
IntelliScaleSetup <- function(data, exclude = c()) {
  scale.info <- data.frame(
    column = colnames(data),
    scaled = rep_len(NA, ncol(data)),
    mean = rep_len(NA, ncol(data)),
    sd = rep_len(NA, ncol(data)))
  
  for (c in 1:ncol(data)) {
    if (colnames(data)[c] %in% exclude) {
      scale.info[c, "scaled"] = F
    } else {
      scale.info[c,"scaled"] = !is.factor(data[,c])
    }
  }
  
  scale.info[scale.info$scaled, "mean"] <- apply(data[,scale.info$scaled], 2, mean)
  scale.info[scale.info$scaled, "sd"] <- apply(data[,scale.info$scaled], 2, sd)
  return(scale.info)
}

IntelliScale <- function(data, scale.info) {
  scaled <- data  
  for (c in 1:ncol(scaled)) {
    scale.info.idx <- which(scale.info$column %in% colnames(scaled)[c])
    if (length(scale.info.idx) == 0) {
      next
    } else if (!scale.info[scale.info.idx, "scaled"]) {
      next
    }
    
    scaled[,c] <- (scaled[,c] - scale.info[scale.info.idx, "mean"]) /
      scale.info[scale.info.idx, "sd"]
  }
  return(scaled)
}
dat.scale.info <- IntelliScaleSetup(dat, exclude=c("price"))
dat.scale <- IntelliScale(dat, dat.scale.info)

# Double-check the scaled values
apply(dat.scale[,dat.scale.info$scaled], 2, mean)
apply(dat.scale[,dat.scale.info$scaled], 2, sd)
colMeans(dat.scale[,c("mileage", "price", "year")])

# 2p cv knn ----
cv.2p.rmse <- RunKKNNCV(price ~ mileage + year, dat.scale, nfolds, k.values)

cv.2p.min.rmse = data.frame(
  k = cv.2p.rmse$k[ which.min(cv.2p.rmse$total) ], 
  rmse = cv.2p.rmse$total[ which.min(cv.2p.rmse$total) ])

# 2p Plot RMSE ----
g <- plot_cv_rmse(cv.2p.rmse, cv.2p.min.rmse, nfolds) +
  theme_bw() + theme(legend.position=c(0.9, 0.25)) +
  scale_x_continuous(expand=c(0.01, 0))

PlotSetup("2p_cv_k")
plot(g)
PlotDone()

# 2p cv knn - multirun ----
nruns <- 10
RunMultiKKNNCV <- function(frm, data, nfolds, nruns, k.values) {
  ncores <- detectCores()
  clust <- makeCluster(ncores, type = "FORK")
  seeds <- sample(1:nruns*100, size = nruns)
  rmse.multi <- parSapply(cl = clust, X = 1:nruns, FUN = function(t) {
    set.seed(seeds[t])
    return(RunKKNNCV(frm, data, nfolds, k.values)$total)
  })
  stopCluster(clust)
  return(data.frame(k=k.values,
                    mean=rowMeans(rmse.multi),
                    run=rmse.multi))
}
cv.2p.rmse.multi <- LoadCachedOrRun(
  RunMultiKKNNCV, 
  price ~ mileage + year, dat.scale, nfolds, nruns, 
  c(seq(10,40,1), seq(45,100,5)))

# Plot 2p cv knn - multirun ----
cv.2p.rmse.multi.melt = melt(cv.2p.rmse.multi, id.vars="k")
plot.colors = gg_color_hue(nruns+1)
g <- ggplot(data=cv.2p.rmse.multi.melt) + 
  geom_point(aes(x=k, y=value, color=variable)) + 
  geom_line(aes(x=k, y=value, color=variable)) +
  scale_color_manual(
    name = "Run",
    values=c(plot.colors[1], alpha(plot.colors[2:(nruns+1)], 0.15)),
    labels = c("Mean", sprintf("Run #%d", 1:nruns))) +
  theme_bw() + labs(y="RMSE")
PlotSetup("2p_cv_multi_k")
plot(g)
PlotDone()

# 2p Compare ----
cv.1p.2p.compare <- data.frame(
  "k" = cv.1p.rmse$k,
  "mileage" = cv.1p.rmse$total,
  "mileage+year" = cv.2p.rmse$total)

cv.1p.2p.rmse.min <- cbind(rbind(cv.1p.min.rmse, cv.2p.min.rmse))
cv.1p.2p.rmse.min$label = sprintf("Minimum @ k=%d", cv.1p.2p.rmse.min$k)
cv.1p.2p.rmse.min.melt <- melt(cv.1p.2p.rmse.min, id.vars=c("k", "label"))

cv.1p.2p.compare.melt <- melt(cv.1p.2p.compare, id.vars="k")
g <- ggplot() + 
  geom_point(data=cv.1p.2p.compare.melt, aes(x=k, y=value, color=variable)) +
  geom_line(data=cv.1p.2p.compare.melt, aes(x=k, y=value, color=variable)) +
  scale_color_discrete("Attributes", 
                       labels=c("Mileage", "Mileage & Year")) +
  geom_point(data=cv.1p.2p.rmse.min.melt, size=5, pch=21, show_guide=F,
             aes(x=k, y=value)) +
  geom_text(data=cv.1p.2p.rmse.min.melt, 
            aes(x=k, y=value, hjust=0, vjust=0,
                label=sprintf("Minimum @ k=%d", cv.1p.2p.rmse.min.melt$k))) +
  theme_bw() + theme(legend.position=c(0.85, 0.85)) +
  labs(x="k", y="RMSE")

PlotSetup("1p_2p_cv_compare")
plot(g)
PlotDone()

# Predictions ----
predict.1p = data.frame(mileage=100e3)
predict.1p$price_hat <- kknn(
  price ~ mileage, train=dat, test=predict.1p, 
  k=cv.1p.min.rmse$k, kernel='rectangular')$fitted.value

ExportTable(predict.1p, "1p_predict", "Predicted Price with 1 Attribute", 
            c("Mileage", "$\\widehat{price}$"), display=c('d', 'd', 'f'), 
            include.rownames=F)

predict.2p = data.frame(year=2008, mileage=75e3)
predict.2p$price_hat <- kknn(
  price ~ mileage + year, train=dat.scale, 
  test=IntelliScale(predict.2p, dat.scale.info), 
  k=cv.2p.min.rmse$k, kernel='rectangular')$fitted.value

ExportTable(predict.2p, "2p_predict", "Predicted Price with 2 Attributes",
            c("Year", "Mileage", "$\\widehat{price}$"), display=c('d', 'd', 'd', 'f'),
            include.rownames=F)

# Exports ----
ExportTable(dat[sample(nrow(dat), 5),], "head_dat",
            "First Few Samples of the Dataset",
            digits=c(0, 0, 0, 0, 0, 0, 0, 0), include.rownames=F)

ExportTable(dat.scale.info[c(4,5), c(1,3,4)], "data_scale",
            "Basic Dataset Statistics", 
            digits=c(0,0,1,1), include.rownames=F)

# How does nfolds affect optimal k? ----

# have this return mean + std
nfolds.seq <- seq(5, 40, 5)
optK <- sapply(1:length(nfolds.seq), function(i) {
  ks <- VarianceKKNN(T = 100, nfolds = nfolds.seq[i])
  return(mean(ks))
})
