rm(list=ls())

require(reshape2)
require(ggplot2)

rx="^train"
file="run_6_layers/loss.csv"

PlotLoss <- function(file, rx) {
    dat <- read.csv(file)
    loss <- dat[, grep(rx, names(dat))]
    loss <- cbind(epoch=dat$epoch, loss)
    melted <- melt(loss, id.vars = 'epoch')
    
    g <- ggplot(data=melted) + 
      geom_line(aes(x=epoch, y=value, color=variable)) +
      scale_y_log10()
    plot(g) 
}