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
    return(as.integer(unlist(strsplit(x, " "))))
  })
dat.raw$Image <- NULL
im.raw <- do.call(rbind, im.raw)

# Show some random faces ######################################################

set.seed(0x0DedBeef)
num.faces <- 8
rand.idx <- sample(1:nrow(dat.raw), num.faces)
pdf("../writeup/random_faces.pdf", width=96*4/72, height=96*2/72)
par(mfrow=c(2,4), mar = c(0,0,0,0), pty="s")
for (i in 1:num.faces) {
  image(matrix(im.raw[rand.idx[i],], 96, 96), 
        col = im.col, xaxt='n', yaxt='n')
  kpx <- dat.raw[rand.idx[i], seq(1, length(avg.kp), 2)]
  kpy <- dat.raw[rand.idx[i], seq(2, length(avg.kp), 2)]
  points(kpx/96, kpy/96, col='red', pch='+')
}
dev.off()

# Visualize accuracy ##########################################################

run_dir = "run_final"

# Average face
avg.face <- matrix(colMeans(im.raw, na.rm=T), 96, 96)

# Average keypoints
avg.kp <- colMeans(dat.raw, na.rm=T)
avg.kpx <- avg.kp[seq(1, length(avg.kp), 2)]
avg.kpy <- avg.kp[seq(2, length(avg.kp), 2)]

features <- unique(gsub("_x|_y", "", names(dat.raw)))
feature.groups <- c("eyebrow", "eye_center", "eye_corner", "mouth_ex_bottom", 
                    "mouth_inc_bottom", "nose")

valid.pred <- read.csv(paste(run_dir, "combined_valid_pred.csv", sep="/"))
valid.actual <- read.csv(paste(run_dir, "combined_valid_actual.csv", sep="/"))

rmse <- sapply(names(dat.raw), function(f) {
  idx.pred <- (valid.pred[, paste("missing", gsub("_x|_y", "", f), sep="_")] < 0.5) &
    (valid.actual[, paste("missing", gsub("_x|_y", "", f), sep="_")] == 0)
  y <- valid.actual[idx.pred, f]
  yhat <- valid.pred[idx.pred, f]
  return(sqrt(mean((y-yhat)^2, na.rm=T)))
})
rmse[order(rmse)]

rmse.tab <- data.frame(name=gsub("_", " ", names(rmse)), rmse=rmse)
rmse.tab <- rmse.tab[order(rmse.tab$rmse),]
ExportTable(table=rmse.tab, file="rmse", 
            caption="RMSE for Predicted Keypoints", 
            colnames = c("Keypoint", "RMSE"), include.rownames=F)

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
radius[order(radius)]

radius.tab <- data.frame(name=gsub("_", " ", names(radius)), radius=radius)
radius.tab <- radius.tab[order(radius.tab$radius),]
ExportTable(table=radius.tab, file="radius", 
            caption="Average Euclidean Distance between Predicted and Actual Keypoints", 
            colnames = c("Keypoint", "Average Distance"), include.rownames=F)

# Colors
pal <- gg_color_hue(6)
pal.light <- add.alpha(pal, alpha=0.2)

# Get feature groups
require(jsonlite)
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

# GOOD AND BAD PREDICTIONS ####################################################

img.rmse <- data.frame(index = valid.pred$index, rmse = rep(NA, nrow(valid.pred)))
img.rmse$rmse <- sapply(1:nrow(valid.pred), function(i) {
  y <- valid.actual[i,2:31]
  yhat <- valid.pred[i,2:31]
  return(sqrt(mean((y-yhat)^2, na.rm=T)))
})

# Restrict to only images where all keypoints are actually there
allthere.idx <- valid.actual$index[rowSums(
  valid.actual[, grep("missing", names(valid.actual))]
) == 0]
img.rmse <- img.rmse[img.rmse$index %in% allthere.idx,]

# Order from most to least accurate
img.rmse <- img.rmse[order(img.rmse$rmse), ]
pdf("../writeup/best_faces.pdf", width=96*3/72, height=96*2/72)
par(mfrow=c(2,3), mar = c(0,0,0,0), pty="s")
for (i in 1:6) {
#   PlotSetup(paste("good_face", i, sep=""))
  
  idx <- img.rmse$index[i]
  # Plot face
  image(matrix(im.raw[idx+1,], 96, 96), 
        col = im.col, xaxt='n', yaxt='n')
  
  # Predicted keypoints
  kpx.pred <- valid.pred[valid.pred$index==idx, grep("_x", names(valid.pred))]
  kpy.pred <- valid.pred[valid.pred$index==idx, grep("_y", names(valid.pred))]
  
  print(paste(kpx.pred[1:3],kpy.pred[1:3]))
  
  # Actual keypoints
  kpx.actual <- valid.actual[valid.actual$index==idx, grep("_x", names(valid.actual))]
  kpy.actual <- valid.actual[valid.actual$index==idx, grep("_y", names(valid.actual))]

  points(kpx.pred/96, kpy.pred/96, col='red', pch='+')
  points(kpx.actual/96, kpy.actual/96, col='green', pch='o')
  
#   PlotDone()
}
dev.off()
mean(img.rmse[1:6,]$rmse) # average rmse

# Order from least to most accurate
img.rmse <- img.rmse[order(img.rmse$rmse, decreasing = T), ]
pdf("../writeup/worst_faces.pdf", width=96*3/72, height=96*2/72)
par(mfrow=c(2,3), mar = c(0,0,0,0), pty="s")
for (i in 1:6) {
#   PlotSetup(paste("bad_face", i, sep=""))
  
  idx <- img.rmse$index[i]
  
  # Plot face
  image(matrix(im.raw[idx+1,], 96, 96), 
        col = im.col, xaxt='n', yaxt='n')
  
  # Predicted keypoints
  kpx.pred <- valid.pred[valid.pred$index==idx, grep("_x", names(valid.pred))]
  kpy.pred <- valid.pred[valid.pred$index==idx, grep("_y", names(valid.pred))]
  
  # Actual keypoints
  kpx.actual <- valid.actual[valid.actual$index==idx, grep("_x", names(valid.actual))]
  kpy.actual <- valid.actual[valid.actual$index==idx, grep("_y", names(valid.actual))]
  
  points(kpx.pred/96, kpy.pred/96, col='red', pch='+')
  points(kpx.actual/96, kpy.actual/96, col='green', pch='o')
  
#   PlotDone()
}
dev.off()
mean(img.rmse[1:6,]$rmse) # average rmse

## Compare to naive averaging
naive.rmse <- sapply(1:ncol(dat.raw), function(c) {
  sqrt(mean((dat.raw[,c] - avg.kp[c])^2, na.rm=T))
})
