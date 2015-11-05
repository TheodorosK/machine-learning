library(xtable)

ExportTable <- function(table, file, caption, colnames=NULL, 
                        align=NULL, digits=NULL, display=NULL,
                        include.rownames=T) {
  if (!is.null(colnames)) { colnames(table) = colnames }
  print(xtable(table, label=paste('tab:', file, sep=''), caption=caption,
               align=align, digits=digits, display=display),
        caption.placement="top",
        sanitize.text.function=function(x){x},
        file=GetFilename(paste(file, '.tex', sep='')),
        include.rownames=include.rownames)
}

ExportWideTable <- function(table, file, caption, colnames=NULL, 
                            align=NULL, digits=NULL, display=NULL,
                            include.rownames=T) {
  if (!is.null(colnames)) { colnames(table) = colnames }
  
  sink(GetFilename(paste(file, '.tex', sep='')))
  cat("\\begin{table}[ht]")
  cat("\\begin{adjustwidth}{-1in}{-1in}")
  cat(print(xtable(table, label=paste('tab:', file, sep=''), caption=caption,
               align=align, digits=digits, display=display),
        caption.placement="top",
        sanitize.text.function=function(x){x},
        file="",
        include.rownames=include.rownames,
        floating = F))
  cat("\\end{adjustwidth}")
  cat("\\end{table}")
  sink()
  
}
