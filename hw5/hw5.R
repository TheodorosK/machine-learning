rm(list=ls())

set.seed(0x0DedBeef)

library(tictoc)
library(distrom)
require(snowfall)
require(caret)
require(doSNOW)

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

# Random forest ----
# In-sample CV accuracy is ~97 percent but OOS is ~94 percent
# If time, try things like regularized random forest

# p <- dim(dat$X_train)[2] = 477; sqrt(p) ~ 22
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

# Boosting tree ----
# Looks like same overfitting issue as RF above

# tune.boost <- expand.grid(interaction.depth = seq(1, 9, 4),
#                           n.trees = seq(500, 2000, 500),
#                           shrinkage = c(0.01, 0.05),
#                           n.minobsinnode = 20)

tune.boost <- expand.grid(interaction.depth = 5, # takes 35 f'ing minutes to run
                          n.trees = 500,
                          shrinkage = c(0.01, 0.05),
                          n.minobsinnode = 20)

ctrl.boost <- trainControl(method = "cv", number = 5, 
                           allowParallel = T)

tic()
model.boost <- train(x = dat$X_train, y = dat$y_train,
                     method = "gbm", trControl = ctrl.boost, tuneGrid = tune.boost)
tt.boost <- toc()

pred.boost <- predict(model.boost, dat$X_test, type = "raw")

sprintf("Boosting tree is %.1f percent accurate (runtime = %.2f mins)", 
        100 * sum(pred.boost == dat$y_test) / length(dat$y_test),
        (tt.boost$toc-tt.boost$tic) / 60)
