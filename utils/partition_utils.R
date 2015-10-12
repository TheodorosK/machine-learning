PartitionDataset <- function(part.sizes, data, reorder = T) {
  if (sum(part.sizes) != 1) {
    stop(sprintf("part.sizes must sum to 1: %f", sum(part.sizes)))
  }
  
  # Reorder the data.
  if (reorder) {
    data <- data[sample(1:nrow(data), nrow(data)),]
  }
  # Calculate the partition indices
  part.indices <- floor(nrow(data) * cumsum(part.sizes))

  partitions <- vector("list", length(part.indices))
  
  last.end.idx <- 0
  part.i <- 1;
  for (part.end.idx in part.indices) {
    partitions[[part.i]] <- data[seq(last.end.idx+1, part.end.idx),]
    last.end.idx <- part.end.idx
    part.i <- part.i + 1
  }
  return(partitions)
}