rm(list=ls())
source('../utils/source_me.R', chdir = T)

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
  return(dat)
}
dat.clean <- CleanData(dat.dirty)

# Partition & Subsampled ----
set.seed(0x0FedBeef)
dat.partitioned <- PartitionDataset(c(0.6, 0.20, 0.20), dat.clean)
dat.subsampled <- SamplePartitionedDataset(dat.partitioned, 0.05)

# Importance Feature Selection ----
require(randomForest)
imp.rf.model <- randomForest(y ~ ., data=dat.subsampled[[1]], importance=T)

# Should we use absolute value here?
imp.rf.model.imp <- importance(imp.rf.model, type=1)
imp.features <- rownames(imp.rf.model.imp)[
  order(imp.rf.model.imp, decreasing = T)]
imp.features <- imp.features[1:10]

varImpPlot(imp.rf.model)

# Extract Important Features ----
ExtractFeatures <- function(data, features) {
  for (i in 1:length(data)) {
    p.dat <- data[[i]]
    data[[i]] <- p.dat[,colnames(p.dat) %in% features]
  }
  return(data)
} 
dat.select <- ExtractFeatures(dat.partitioned, c(imp.features, "y"))

# Random Forest Model ----
rf.model <- randomForest(y ~ ., data=dat.select[[1]], do.trace=T, ntree=500, mtry=10)
