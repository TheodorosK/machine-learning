rm(list=ls())

library(tictoc)
library(distrom)

source('../utils/source_me.R', chdir = T)
source('../utils/parse_data.R', chdir = T)
CreateDefaultPlotOpts(WriteToFile = T)

dat <- parse_human_activity_recog_data()

# Distributed multinomial regression ----

cl <- makeCluster(detectCores())
tic()
model.dmr <- dmr(cl = cl, covars = dat$X_train, counts = dat$y_train, 
                 cv = T, verb = T)
toc()
stopCluster(cl)

pred.dmr <- predict(model.dmr, dat$X_test, type = "response")
