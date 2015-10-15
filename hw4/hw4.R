rm(list=ls())
source('../utils/source_me.R', chdir = T)

LoadData <- function(what) {
  data.dir <- "../data/kdd_2009"
  prefix <- "orange_small_train"
  y <- read.delim(file.path(data.dir, sprintf("%s_%s.labels.txt", prefix, what)), 
                  col.names="y", header = F)
  x <- read.delim(file.path(data.dir, sprintf("%s.data.gz", prefix)))
  return(cbind(y, x))
}
rawDat <- LoadData("churn")
# dat <- PartitionDataset(c(0.5, 0.25, 0.25), rawDat)

# Clean Data ----
CleanData <- function(dat) {
  na.count.by.var <- apply(dat, 2, function(x) sum(is.na(x)))
  all.na.vars <- which(na.count.by.var == nrow(dat))
  print(sprintf("Removing columns containing only NAs: %s", 
                paste(colnames(dat)[all.na.vars], collapse = ",")))
  dat <- dat[,-all.na.vars]
  
  return(dat)
}
dat <- CleanData(rawDat)
frac.levels <- unlist(lapply(dat, nlevels))/nrow(dat)
frac.levels[which(frac.levels != 0)]
