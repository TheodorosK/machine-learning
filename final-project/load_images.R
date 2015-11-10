rm(list = ls())

source("../utils/source_me.R", chdir = T)
CreateDefaultPlotOpts()
Global.PlotOpts$Prefix <- "proposal/"

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
dat.raw <- LoadCacheTagOrRun("raw", read.csv, "data/training.csv",
                             stringsAsFactors=F)
im.raw <- LoadCacheTagOrRun(
  "raw_im", sfLapply,
  dat.raw[,"Image"], function(x) {
    return(matrix(rev(as.integer(unlist(strsplit(x, " ")))), 96, 96))
  })
dat.raw$Image <- NULL

# Do a couple of sanity checks.
print(sprintf("# of images == # of data-points: %s", 
              length(im.raw) == nrow(dat.raw)))
image(im.raw[[1]], col=im.col)
image(im.raw[[2]], col=im.col)

# Cleanup Images ##############################################################
source("image_processing.R")

if (sfParallel()) {
  sfRemoveAll()
  sfExport(list=c("NormalizeFilter", "NormalizeImage", "Convolve"))
}  
im.edge_det <- LoadCacheTagOrRun(
  'edge_det_im', sfLapply, 
  im.raw, function(x, out.min=0, out.max=255) {
    # Here are some basic Edge-detection kernels:
    # https://en.wikipedia.org/wiki/Kernel_(image_processing)
    #
    # I want to try this at some point (it looks cool and does a good 
    # job rejecting noise):
    # https://en.wikipedia.org/wiki/Canny_edge_detector
    #
    # filt <- matrix(c(1, 0, -1, 0, 0, 0, -1, 0, 1), 3, 3)
    filt <- matrix(c(-1, -1, -1, -1, 8, -1, -1, -1, -1), 3, 3)
    conv <- Convolve(x, filt)
    # Saturate before Normalization
    conv[conv < out.min] <- out.min
    conv[conv > out.max] <- out.max
    return(matrix(as.integer(round(conv)), nrow(conv), ncol(conv)))
  })

layout(matrix(1:6, 2, 3))
for (i in sample(1:length(im.edge_det), 3)) {
  image(im.edge_det[[i]], col=im.col, main=i)
  image(im.raw[[i]], col=im.col, main=i)
}

# Sobel Edge Detection ########################################################
if (sfParallel()) {
  sfRemoveAll()
  sfExport(list=c("NormalizeFilter", "NormalizeImage", "Convolve"))
}  
im.sobel_det <- LoadCacheTagOrRun(
  "sobel_edge_im", 
  sfLapply, x=im.raw, function(x) {
  # Sobel Operator:
  # https://en.wikipedia.org/wiki/Sobel_operator
  # Sobel Kernel:
  #   sobel.kern <- matrix(c(-1, 0, 1, -2, 0, 2, -1, 0, 1), 3, 3)
  # Sobel-Feldman Kernel:
  sobel.kern <- matrix(c(3, 10, 3, 0, 0, 0, -3, -10, -3), 3, 3)
  Gx <- Convolve(x, sobel.kern)
  Gy <- Convolve(x, t(sobel.kern))
  G <- sqrt(Gx^2 + Gy^2)
  return(NormalizeImage(G))
})

layout(matrix(1:9, 3, 3))
for (i in sample(1:length(im.sobel_det), 3)) {
  image(im.sobel_det[[i]], col=im.col, main=i)
  image(im.edge_det[[i]], col=im.col, main=i)
  image(im.raw[[i]], col=im.col, main=i)
}

# Display images for proposal #################################################

# R assumes 72 pixels / inch by default
pdf(file = GetFilename('sample_faces.pdf'), width = 96/72*3, height = 96/72*3)
par(mfcol=c(3,3), pty="s", mar=c(0.1,0.1,0,0)+0.1)
for (i in sample(length(im.raw), 3)) {
  image(im.raw[[i]], col=im.col, xaxt='n', yaxt='n')
  image(im.edge_det[[i]], col=im.col, xaxt='n', yaxt='n')
  image(im.sobel_det[[i]], col=im.col, xaxt='n', yaxt='n')
}
dev.off()


