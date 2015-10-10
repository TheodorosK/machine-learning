# HW3

# Clear Vars & Load Libraries ----
rm(list = ls())

library(tree)

source('../utils/source_me.R', chdir = T)
CreateDefaultPlotOpts(WriteToFile = T)

# Cars data set: goal is to predict price
cars <- read.csv(file = "../data/usedcars.csv")

# Trees ----

tree.small = tree(price ~ mileage, data = cars)
plot(tree.small)
text(tree.small)

tree.large = tree(price ~ ., data = cars)
plot(tree.large)
text(tree.large)
