rm(list=ls())

set.seed(0x0DedBeef)

library(tictoc)
library(distrom)
require(snowfall)
require(caret)
require(doSNOW)
library(ggplot2)
library(reshape2)

sfInit(cpus=detectCores(), parallel=T)
registerDoSNOW(sfGetCluster())

source('../utils/source_me.R', chdir = T)
source('../utils/parse_data.R', chdir = T)
CreateDefaultPlotOpts(WriteToFile = T)

sfInit(cpus=detectCores(), parallel=T)
registerDoSNOW(sfGetCluster())

dat <- parse_human_activity_recog_data()

# Distributed multinomial regression ----

cl <- makeCluster(detectCores())
tic()
model.dmr <- dmr(cl = cl, covars = dat$X_train, counts = dat$y_train, 
                 cv = T, verb = T)
tt.dmr <- toc()
stopCluster(cl)

pred.dmr <- predict(model.dmr, dat$X_test, type = "class")

sprintf("Distributed multinomial regression is %.1f percent accurate (runtime = %.2f mins)", 
        100 * sum(pred.dmr == dat$y_test) / length(dat$y_test),
        (tt.dmr$toc-tt.dmr$tic) / 60)

conmat.dmr <- confusionMatrix(pred.dmr, dat$y_test)

# Random forest ----
# In-sample CV accuracy is ~97 percent but OOS is ~94 percent
# If time, try things like regularized random forest

# p <- dim(dat$X_train)[2] = 477; sqrt(p) ~ 22
# The final value used for the model was mtry = 10. 
tune.rf <- expand.grid(mtry = seq(10, 30, 5)) # takes about 10 mins on 8 cores

ctrl.rf <- trainControl(method = "cv", number = 5, 
                        allowParallel = T)

tic()
model.rf <- train(x = dat$X_train, y = dat$y_train, ntree = 500,
                  method = "rf", trControl = ctrl.rf, tuneGrid = tune.rf)
tt.rf <- toc()

pred.rf <- predict(model.rf, dat$X_test, type = "raw")

sprintf("Random forest is %.1f percent accurate (runtime = %.2f mins)", 
        100 * sum(pred.rf == dat$y_test) / length(dat$y_test),
        (tt.rf$toc-tt.rf$tic) / 60)

conmat.rf <- confusionMatrix(pred.rf, dat$y_test)

# Boosting tree ----
# Looks like same overfitting issue as RF above

# The final values used for the model were n.trees = 2000, interaction.depth = 5, shrinkage =
#   0.05 and n.minobsinnode = 20.
tune.boost <- expand.grid(interaction.depth = seq(1, 9, 4), # takes 4.3 hours
                          n.trees = seq(500, 2000, 500),
                          shrinkage = c(0.01, 0.05),
                          n.minobsinnode = 20)

# tune.boost <- expand.grid(interaction.depth = 5, # takes 35 f'ing minutes to run
#                           n.trees = 500,
#                           shrinkage = c(0.01, 0.05),
#                           n.minobsinnode = 20)

ctrl.boost <- trainControl(method = "cv", number = 5, 
                           allowParallel = T)

tic()
model.boost <- train(x = dat$X_train, y = dat$y_train,
                     method = "gbm", trControl = ctrl.boost, tuneGrid = tune.boost)
tt.boost <- toc()

pred.boost <- predict(model.boost, dat$X_test, type = "raw")

sprintf("Boosting tree is %.1f percent accurate (runtime = %.2f hours)", 
        100 * sum(pred.boost == dat$y_test) / length(dat$y_test),
        (tt.boost$toc-tt.boost$tic) / 3600)

conmat.boost <- confusionMatrix(pred.boost, dat$y_test)

# Try re-shuffling the data ----

dat.all <- cBind(as.factor(c(as.character(dat$y_train), as.character(dat$y_test))),
                 rbind(dat$X_train, dat$X_test))
names(dat.all)[1] <- "y"

dat.repart <- PartitionDataset(c(0.7, 0.3), dat.all)
redat <- list(dat.repart[[1]][, -1, with=F],
              dat.repart[[2]][, -1, with=F],
              dat.repart[[1]]$y,
              dat.repart[[2]]$y)
names(redat) <- c("X_train", "X_test", "y_train", "y_test")

# Random forest (use optimal params from above)
if (sfParallel()) {
  sfRemoveAll()
  sfExport("redat")
}
tic()
model.rf.repart <- foreach(ntree=rep(125, sfCpus()), .combine=combine,
                           .packages='randomForest') %dopar%
  randomForest(x = redat$X_train, y = redat$y_train, mtry = 10, do.trace=T, importance=T)
tt.rf.repart <- toc()

pred.rf.repart <- predict(model.rf.repart, redat$X_test, type = "response")

sprintf("Re-shuffled random forest is %.1f percent accurate (runtime = %.2f mins)", 
        100 * sum(pred.rf.repart == redat$y_test) / length(redat$y_test),
        (tt.rf.repart$toc-tt.rf.repart$tic) / 60)

# Stop cluster ----

sfStop()

# Deep learning ----

library(h2o)
h2o.init(nthreads = -1)

p <- ncol(dat$X_train)
y.nlevels <- nlevels(dat$y_train)


dat.h2o <- list(X_train = as.h2o(dat$X_train, destination_frame = "X_train"),
                y_train = as.h2o(dat$y_train, destination_frame = "y_train"),
                X_test = as.h2o(dat$X_test, destination_frame = "X_test"),
                y_test = as.h2o(dat$y_test, destination_frame = "y_test"))

model.nn1 <- h2o.deeplearning(x = 1:p, y = p+1, 
                              training_frame = h2o.cbind(dat.h2o$X_train, dat.h2o$y_train),
                              activation = "TanhWithDropout",
                              input_dropout_ratio = 0.50,
                              hidden = c(p+y.nlevels, p+y.nlevels),
                              hidden_dropout_ratios = c(0.5, 0.5),
                              l1 = 1e-5,
                              epochs = 50,
                              model_id = "model.nn1")
pred.nn1 <- as.data.frame(h2o.predict(model.nn1, dat.h2o$X_test))
conmat.nn1 <- confusionMatrix(pred.nn1$predict, dat$y_test)
print(conmat.nn1$overall[1])

# Visualization ----

ConfusionHeatMap(conmat.dmr, title="Confusion Matrix Heatmap: Multinomial Logit", fname="heatmap_dmr")
ConfusionHeatMap(conmat.rf, title="Confusion Matrix Heatmap: Random Forest", fname="heatmap_rf")
ConfusionHeatMap(conmat.boost, title="Confusion Matrix Heatmap: Boosting Tree", fname="heatmap_boost")

export.cols <- c("Sensitivity", "Specificity", "Balanced Accuracy")
ExportTable(table = conmat.dmr$byClass[,c("Sensitivity", "Specificity", "Balanced Accuracy")],
            file = "conmat_dmr", 
            caption = "Confusion Matrix for Multinomial Logit Regression", 
            digits = 2, 
            colnames = c("Sensitivity", "Specificity", "Balanced Accuracy"), 
            include.rownames = T)

