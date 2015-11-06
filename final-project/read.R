rm(list = ls())

source("../utils/source_me.R", chdir = T)
CreateDefaultPlotOpts()

require(doSNOW)
require(snowfall)
require(parallel)

sfInit(cpus=detectCores(), parallel = T)
registerDoSNOW(sfGetCluster())

# Load Data ###################################################################
raw_dat <- LoadCacheTagOrRun("raw_data", read.csv, "data/training.csv",
                             stringsAsFactors=F)
raw_images <- LoadCacheTagOrRun(
  "raw_images", sfLapply,
  raw_dat[,"Image"], function(x) {
    return(matrix(rev(as.integer(unlist(strsplit(x, " ")))), 96, 96))
  })
raw_dat$Image <- NULL

# Do a couple of sanity checks.
print(sprintf("# of images == # of data-points: %s", 
              length(raw_images) == nrow(raw_dat)))
image(raw_images[[1]], col=gray((0:255)/255))
image(raw_images[[2]], col=gray((0:255)/255))

# Cleanup Images ##############################################################
