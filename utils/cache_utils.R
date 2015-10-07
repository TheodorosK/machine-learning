library(digest)

# id should be unique for the entire project (i.e. directory) and should only
# contain letters/numbers that you'd want in a filename (i.e. no spaces).
LoadCachedOrRun <- function(FUN, ..., FORCE.RUN = F) {
  FUN <- match.fun(FUN)
  cache.digest <- digest(c(FUN, ...), algo="md5")
  
  dir.create("cached", showWarnings = F)
  
  datFile <- sprintf("cached/%s.dat", cache.digest)
  if (file.exists(datFile) && !FORCE.RUN) {
    warning(sprintf("NOTE: Loading Cached Datastore '%s'", cache.digest))
    load(datFile, verbose = T)
  } else {
    warning(sprintf("NOTE: Populating Cached Datastore '%s'", cache.digest))
    data <- FUN(...)
    save(list=c("data"), file=datFile)
  }
  return(data)
}

ClearAllCachedRuns <- function() {
  unlink("cached", recursive = T)
}