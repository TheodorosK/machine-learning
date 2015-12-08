rm(list=ls())

source("../../utils/source_me.R", chdir = T)
CreateDefaultPlotOpts()
Global.PlotOpts$Prefix <- "../writeup/"

require(doSNOW)
require(snowfall)
require(parallel)

if (!sfIsRunning()) {
  sfInit(cpus=detectCores(), parallel = T)
  registerDoSNOW(sfGetCluster())
}

# We seem to use this color palette everywhere, just define it once
im.col <- gray((0:255)/255)

# Load Data ###################################################################
dat.raw <- LoadCacheTagOrRun("raw", read.csv, "../data/training.csv",
                             stringsAsFactors=F)
im.raw <- LoadCacheTagOrRun(
  "raw_im", sfLapply,
  dat.raw[,"Image"], function(x) {
    return(rev(as.integer(unlist(strsplit(x, " ")))))
  })
dat.raw$Image <- NULL
im.raw <- do.call(rbind, im.raw)

# Visualize ###################################################################

# Average face
avg.face <- matrix(colMeans(im.raw, na.rm=T), 96, 96)

# Average keypoints
avg.kp <- colMeans(dat.raw, na.rm=T)
avg.kpx <- avg.kp[seq(1, length(avg.kp), 2)]
avg.kpy <- 96-avg.kp[seq(2, length(avg.kp), 2)]

features <- unique(gsub("_x|_y", "", names(dat.raw)))
feature.groups <- c("eyebrow", "eye_center", "eye_corner", "mouth_ex_bottom", 
                    "mouth_inc_bottom", "nose")

valid.pred <- read.csv("run_final/combined_valid_pred.csv")
valid.actual <- read.csv("run_final/combined_valid_actual.csv")

rmse <- sapply(names(dat.raw), function(f) {
  idx.pred <- (valid.pred[, paste("missing", gsub("_x|_y", "", f), sep="_")] < 0.5) &
    (valid.actual[, paste("missing", gsub("_x|_y", "", f), sep="_")] == 0)
  y <- valid.actual[idx.pred, f]
  yhat <- valid.pred[idx.pred, f]
  return(sqrt(mean((y-yhat)^2, na.rm=T)))
})

radius <- sapply(features, function(f) {
  idx.pred <- (valid.pred[, paste("missing", f, sep="_")] < 0.5) &
    (valid.actual[, paste("missing", f, sep="_")] == 0)
  x <- valid.actual[idx.pred, paste(f, "x", sep="_")]
  xhat <- valid.pred[idx.pred, paste(f, "x", sep="_")]
  y <- valid.actual[idx.pred, paste(f, "y", sep="_")]
  yhat <- valid.pred[idx.pred, paste(f, "y", sep="_")]
  radius <- mean(sqrt((x-xhat)^2+(y-yhat)^2))
  return(radius)
})

# Colors
pal <- gg_color_hue(6)
pal.light <- add.alpha(pal, alpha=0.2)

# Get feature groups
require(jsonlite)
feature_groups.json
fgps <- fromJSON("feature_groups.json")

# Plot
PlotSetup("avg_face_rmse")
image(avg.face, col = im.col, xaxt='n', yaxt='n')
for (i in 1:length(fgps)) {
  fn <- unique(gsub("_x|_y", "", names(dat.raw)[fgps[[i]]+1]))
  cat(fn, "\n\n")
  points(avg.kpx[paste(fn, "x", sep="_")]/96, 
         avg.kpy[paste(fn, "y", sep="_")]/96, col=pal[i], pch='+')  
  symbols(avg.kpx[paste(fn, "x", sep="_")]/96, 
          avg.kpy[paste(fn, "y", sep="_")]/96, 
          circles=radius[fn]/96, 
          fg=pal[i], bg=pal.light[i],
          inches=F, add=T)
}
PlotDone()

# SUCCESSES AND BIG MISSES ####################################################

img.rmse <- data.frame(index = valid.pred$index, rmse = rep(NA, nrow(valid.pred)))
img.rmse$rmse <- sapply(1:nrow(valid.pred), function(i) {
  y <- valid.actual[i,2:31]
  yhat <- valid.pred[i,2:31]
  return(sqrt(mean((y-yhat)^2, na.rm=T)))
})

img.rmse <- img.rmse[order(img.rmse$rmse), ]

i <- 1
tmp.face <- matrix(im.raw[img.rmse$index[i],], 96, 96)
image(tmp.face, col = im.col, xaxt='n', yaxt='n')
tmp.kpx <- valid.pred[valid.pred$index==img.rmse$index[i],]
points(valid.pred[valid.pred$index==img.rmse$index[i], seq(2, 30, 2)]/96, 
       (96-valid.pred[valid.pred$index==img.rmse$index[i], seq(3, 31, 2)])/96, 
       col='red', pch='+')  
points(valid.actual[valid.actual$index==img.rmse$index[i], seq(2, 30, 2)]/96, 
       (96-valid.actual[valid.actual$index==img.rmse$index[i], seq(3, 31, 2)])/96, 
       col='red', pch='o')  

