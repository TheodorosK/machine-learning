##### HW1
#
# Setup ----
#
rm(list = ls())

library(ggplot2)
library(kknn)
library(cvTools)
library(reshape2)

set.seed(926)

#
# Read data ----
#
cars <- read.csv("../data/susedcars.csv")

# Visualize
lin <- glm(price ~ mileage, data = cars)

g <- ggplot() + geom_point(data=cars, aes(x=mileage/1000, y=price/1000)) +
  geom_abline(aes(intercept = lin$coefficients[1]/1000,
                  slope = lin$coefficients[2]), color="red", size=1.25) +
  labs(x="Mileage [1000 miles]", y="Price [1000 $]") +
  ggtitle("Linear Regression of Price on Mileage")
print(g)

#
# Knn (No CV) ----
#
nTrain <- round(0.9 * nrow(cars))
tInd <- sample(1:nrow(cars), nTrain)

cars.train <- cars[tInd, ]
cars.test <- cars[-tInd, ]

# estimate RMSE for all possible values of k
nearestNeighbors <- 2:100

out <- data.frame(nn = nearestNeighbors,
                  kknnRmse = rep_len(NA, length(nearestNeighbors)),
                  cvRmse = rep_len(NA, length(nearestNeighbors)))

out$kknnRmse <- sapply(out$nn, function(k) {
  cat(sprintf("%d,", k))
  kn <- kknn(price ~ mileage, train = cars.train, test = cars.test, k = k)
  rmse <- sqrt(mean((cars.test$price - kn$fitted.values)^2))
  return(rmse)
})
cat("done.\n")

print(ggplot(data=out, aes(nn, kknnRmse)) + geom_point() + geom_line() +
        labs(x="Nearest Neighbors", y="RMSE"))

print(sprintf("kknn Min RMSE=%3.2f (@%d nearest neighbors)", 
              min(out$kknnRmse), out$nn[which.min(out$kknnRmse)]))

#
# KNN (CV) ----
#

# This is the packaged way to do it, but the documentation is terrible
# "cv.kknn performs k-fold crossvalidation and is generally slower and does not
# yet contain the test of different models yet"
kn.cv <- cv.kknn(price ~ mileage, data = cars, kcv = 10)

# Home-made way to do k-fold cross validation

nfolds <- 10

folds <- cvFolds(n = nrow(cars), K = nfolds)

out$cvRmse <- sapply(out$nn, function(k) {
  cat(sprintf("%d,", k))
  
  # calculate CV error
  cvErr <- sapply(1:nfolds, function(f) {
    
    testInd <- folds$subsets[folds$which == f]
    trainInd <- folds$subsets[folds$which != f]
    
    cars.test <- cars[testInd, ]
    cars.train <- cars[trainInd, ]
    
    kn <- kknn(price ~ mileage, train = cars.train, test = cars.test, k = k)
    rmse <- sqrt(mean((cars.test$price - kn$fitted.values)^2))
    return(rmse)
  })
  
  return(mean(cvErr))
  
})
cat("done.\n")


#
# Visualize RMSE ----
#
algMins <- data.frame(algo=c("kknn", "cv"), 
                      nn = c(out$nn[which.min(out$kknnRmse)],
                             out$nn[which.min(out$cvRmse)]),
                      rmse = c(min(out$kknnRmse), min(out$cvRmse)))

# Plot the RMSE for difference # of Nearest Neighbors
library(dplyr)
library(reshape2)
out.me <- melt(out, id.vars = "nn")
out.mins.me <- out.me %>% group_by(variable) %>% mutate(shape = min(value) == value)

g <- ggplot(data=out.me, aes(x=nn, y=value, color=variable)) +
  geom_line() + geom_point() + 
  scale_color_discrete("Method", labels=c("kknn\n (OOS - 10% holdout)", "kknn\n (CV - 10-fold)")) +
  labs(x="# of Nearest Neighbors", y="RMSE") +
  geom_point(data=out.mins.me, aes(shape=shape), size=5, na.rm = T, show_guide=F) +
  scale_shape_manual(values = c(NA, 1)) +
  ggtitle("RMSE vs. # of Nearest Neighbors for KKNN")
print(g)

# Plot various k values ----
fittedK <- function(k, test) {
  kn <- kknn(price ~ mileage, train = cars.train, test = test, k = k)
  return(kn$fitted.values)
}

cars.test.sorted <- cars.test[order(cars.test$mileage),]
kValues <- sort(c(2, 400, 40))
fitted <- data.frame(mileage = cars.test.sorted$mileage,
                     sapply(kValues, test=cars.test.sorted, FUN=fittedK))

# Legend still has slashes despite my best efforts with guides. :(
# wclark3: show_guide = F fixes this
#   also switched to scale_color_manual so I could make the linear fit black
#   (thought the default coloration was hard to look at)

# Replicates the ggplot2 color palette
# http://stackoverflow.com/questions/8197559/emulate-ggplot2-default-color-palette
gg_color_hue <- function(n) {
  hues = seq(15, 375, length=n+1)
  hcl(h=hues, l=65, c=100)[1:n]
}

g <- ggplot() + geom_point(data=cars, aes(x=mileage/1000, y=price/1000), alpha=0.3) +
  geom_line(data=melt(fitted, id.vars="mileage"), 
            aes(x=mileage/1000, y=value/1000, color=variable), size=1.25) +
  geom_abline(aes(intercept = lin$coefficients[1]/1000,
                  slope = lin$coefficients[2], color="break"), size=0.75, lty="dashed") +
  scale_color_manual(values = c("#000000", gg_color_hue(3)),
                     name = "Algorithm",
                     labels = c("Linear", sapply(kValues, function(k) sprintf("kknn\n(k=%d)", k)))) +
  guides(shape=guide_legend(override.aes = list(linetype = 0))) + 
  labs(x="Mileage [1000 miles]", y="Price [1000 $]") +
  ggtitle("Predictive Models for Car Price vs. Mileage") + 
  theme_bw() + theme(legend.position = c(0.9, 0.75))
plot(g)