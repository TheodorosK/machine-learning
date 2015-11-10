# Some Basic Image Processing Functions
#
# I Couldn't find a good project, so I just created these.  In theory 
# they're correct, not sure about practice though :)
#
Convolve <- function(image, filter) {
  # Flip filter LR/UD
  filter <- matrix(rev(filter), nrow(filter), ncol(filter))
  num.row <- nrow(image) - nrow(filter) + 1
  num.col <- ncol(image) - ncol(filter) + 1
  new.image <- matrix(rep_len(NA, num.row * num.col), num.row, num.col)
  for (i in 1:num.row) {
    for (j in 1:num.col) {
      new.image[i,j] <- sum(
        image[i:(i+nrow(filter)-1), j:(j+ncol(filter)-1)] * filter)
    }
  }
  return(new.image)
}
NormalizeFilter <- function(raw_filter) {
  return(raw_filter / sum(abs(raw_filter)))
}
NormalizeImage <- function(image, out.min = 0, out.max = 255) {
  image <- (image - min(image) + out.min) *
    ((out.max - out.min) / (max(image) - min(image)))
  return(matrix(as.integer(round(image)), nrow(image), ncol(image))) 
}
