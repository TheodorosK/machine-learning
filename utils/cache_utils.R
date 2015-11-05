library(digest)

LoadCacheTagOrRun <- function(tag, FUN, ..., FORCE.RUN = F) {
  FUN <- match.fun(FUN)
  dir.create("cached", showWarnings = F)
  
  datFile <- sprintf("cached/%s.dat", tag)
  if (file.exists(datFile) && !FORCE.RUN) {
    warning(sprintf("NOTE: Loading Cached Datastore '%s'", tag))
    load(datFile, verbose = T)
  } else {
    warning(sprintf("NOTE: Populating Cached Datastore '%s'", tag))
    data <- FUN(...)
    save(list=c("data"), file=datFile)
  }
  return(data)
}

# id should be unique for the entire project (i.e. directory) and should only
# contain letters/numbers that you'd want in a filename (i.e. no spaces).
LoadCachedOrRun <- function(FUN, ..., FORCE.RUN = F) {
  FUN <- match.fun(FUN)
  cache.digest <- digest(c(FUN, ...), algo="md5")
  return(LoadCacheTagOrRun(cache.digest, FUN, ...))
}

ClearAllCachedRuns <- function() {
  unlink("cached", recursive = T)
}