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
CleanData <- function(dat, na.exclude.threshold.pcnt = 0.95, 
                      many.factor.threshold = 20,
                      level.thresholds = data.frame(
                        name = c("low", "medium", "high"),
                        lower = c(0, 250, 500),
                        upper = c(249, 499, 999))) {
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
    level.count <- table(dat[,c])     
    for (i in 1:nrow(level.thresholds)) {
      to.replace <- level.count >= level.thresholds[i, "lower"] &
        level.count <= level.thresholds[i, "upper"]
      levels(dat[,c])[to.replace] <- as.character(level.thresholds[i, "name"])
    }
    print(sprintf("Column %d - #levels before=%6d, after=%6d", 
                  c, before.nlevels, nlevels(dat[,c])))
  }
  return(dat)
}
dat.clean <- CleanData(dat.dirty)

# Partition ----
dat.partitioned <- PartitionDataset(
  c(0.6, 0.20, 0.20), dat.clean, subsample.amount = 0.05)
dat.train <- dat.partitioned[[1]]
dat.valid <- dat.partitioned[[2]]
# dat.test <- dat.partitioned[[3]]

