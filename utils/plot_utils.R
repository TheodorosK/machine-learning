# Some plot utilities to help create png files with consistent sizes, etc.
#
# To output a plot to a file do the following:
#
# # Create the file
# PlotSetup('filename')
#
# # Do the actual plotting
# plot(...)
#
# # Close the output file
# PlotDone()
#
# This will create a file: 'output/filename.png' in	the current working dir.
#
# NOTE: it is the	caller's responsibility	to ensure that the filename does not
# collide	with other files in the	output directory.  No warning will be shown 
# if the file already exists.


# Appends the arguments to the prefix, also creates an output directory (recursively)
# from the current working directory.
GetFilename <- function(filename, plotOpts. = plotOpts) {
  dir.create(path = plotOpts.$Prefix,
             showWarnings = F, recursive = T)
  return(paste(plotOpts.$Prefix, filename, sep = ''))
}

# Call this before plotting, note, name should be unique so that it is not overwritten
PlotSetup <- function(filename, plotOpts. = plotOpts, usePng = F) {
  if (OutputToFile == F) return()
  
#     setEPS()
#     postscript(GetFilename(paste(filename, 'eps', sep = '.'), plotOpts = plotOpts.))
  if (usePng) {
    png(file = GetFilename(paste(filename, 'png', sep = '.'), plotOpts = plotOpts.),
        width = plotOpts.$Width, height = plotOpts.$Height,
        units = plotOpts.$Units,
        pointsize = plotOpts.$PointSize,
        res = plotOpts.$Res)
  } else {
    pdf(file = GetFilename(paste(filename, 'pdf', sep = '.'), plotOpts = plotOpts.),
        pointsize = plotOpts.$PointSize, compress=T)
  }
}

# Call this after plotting
PlotDone <- function() {
  if (OutputToFile == F) return()
  
  dev.off()
}
