library(xtable)
ExportTable <- function(table, file, caption, colnames=NULL, 
                        align=NULL, digits=NULL, display=NULL) {
  if (!is.null(colnames)) { colnames(table) = colnames }
  print(xtable(table, label=paste('tab:', file, sep=''), caption=caption,
               align=align, digits=digits, display=display),
        sanitize.text.function=function(x){x},
        file=GetFilename(paste(file, '.tex', sep='')))
}
