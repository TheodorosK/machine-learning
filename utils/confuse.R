require(xtable)

write.confusionMatrix <- function(mtx, file, caption) {
  mtx <- as.data.frame.matrix(mtx)
  mtx <- cbind(Predicted=rownames(mtx), mtx)
  addtorow <- list(
    pos=list(-1),
    command=c(sprintf("&\\multicolumn{%d}{c}{Reference}\\\\\n", nrow(mtx))))
  print(add.to.row=addtorow, include.rownames=F, 
        caption.placement="top",
        sanitize.text.function=function(x){x},
        file=GetFilename(paste(file, '.tex', sep='')),
        xtable(mtx, align=c("l", "l|", rep_len("r", ncol(mtx)-1)),
               label=paste('tab:', file, sep=''), caption=caption))
}