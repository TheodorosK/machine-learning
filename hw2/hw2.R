# HW2

# Setup & Load Data ----
rm(list = ls())

source('../utils/source_me.R', chdir = T)
CreateDefaultPlotOpts(WriteToFile = F)

dat <- read.csv(file="../data/susedcars.csv", header = T)

