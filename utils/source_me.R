# OutputToFile is a global flag that enables
OutputToFile = T

# Creates the plotOpts data frame, and an output directory based on the current working
# directory.
plotOpts <- data.frame(Prefix = "output/", Width = 7, Height = 7, Units = "in",
                       Res = 300, PointSize = 12, stringsAsFactors = FALSE)

# Source the plot utilities
source('plot_utils.R')
