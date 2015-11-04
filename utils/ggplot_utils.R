# Replicates the ggplot2 color palette
# http://stackoverflow.com/questions/8197559/emulate-ggplot2-default-color-palette
gg_color_hue <- function(n) {
  hues = seq(15, 375, length=n+1)
  hcl(h=hues, l=65, c=100)[1:n]
}

# Makes a heatmap out of a confusion matrix (from package caret)
ConfusionHeatMap <- function(conmat, title = "Confusion Matrix Heat Map", fname=NA) {
  conmat.melt <- melt(conmat$table/colSums(conmat$table))
  p <- ggplot(conmat.melt, aes(x=Prediction, y=Reference, fill=value)) + 
    geom_tile() + 
    scale_fill_gradient(low="white", high="black") + 
    ggtitle(title) + 
    theme(legend.position = c(0, 1), 
         legend.justification = c(0, 1), 
         legend.background = element_rect(colour = NA, fill = "white"))
  if (is.na(fname)) {
    plot(p)
  } else {
    PlotSetup(fname)
    print(p)
    PlotDone()
  }
}