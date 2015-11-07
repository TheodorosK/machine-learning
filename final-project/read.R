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

# Display images for proposal #################################################

# R assumes 72 pixels / inch by default
pdf(file = GetFilename('sample_faces.pdf'), width = 96/72*3, height = 96/72)
par(mfrow=c(1,3), pty="s", mar=c(0.1,0.1,0,0)+0.1)
for (i in sample(length(raw_images), 3)) {
  image(raw_images[[i]], col=gray((0:255)/255), xaxt='n', yaxt='n')  
}
dev.off()

# Cleanup Images ##############################################################
