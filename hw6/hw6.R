# RESET! ######################################################################
rm(list = ls())

require(jsonlite)
require(recommenderlab)

source("../utils/source_me.R", chdir=T)
CreateDefaultPlotOpts()

# Load Data ###################################################################
fileConnection <- gzcon(url("https://github.com/ChicagoBoothML/MachineLearning_Fall2015/raw/master/Programming%20Scripts/Lecture07/hw/videoGames.json.gz"))
dat <- LoadCacheTagOrRun('raw', stream_in, fileConnection)
close(fileConnection)

# Cleanup #####################################################################

# create a ratingData matrix using reviewerID, itemID, and rating
ratingData <- as(dat[c("reviewerID", "itemID", "rating")], "realRatingMatrix")

# we keep users that have rated more than 2 video games
ratingData <- ratingData[rowCounts(ratingData) > 2,]

# we will focus only on popular video games that have 
# been rated by more than 3 times
ratingData <- ratingData[,colCounts(ratingData) > 3]

# we are left with this many users and items
dim(ratingData)

# Example #####################################################################

# example on how to recommend using Popular method
# r <- Recommender(ratingData, method="Popular")

# recommend 5 items to user it row 10
# rec <- predict(r, ratingData[10, ], type="topNList", n=5)
# as(rec, "list")

# predict ratings 
# rec <- predict(r, ratingData[10, ], type="ratings")
# as(rec, "matrix")

# Question 1: Which user has rated the most games? ############################

userMost <- which.max(rowCounts(ratingData))
rowCounts(ratingData[userMost, ])

# Question 2: Which game has been rated by the most users? ####################

gameMost <- which.max(colCounts(ratingData))
colCounts(ratingData[, gameMost])

# Question 3: Which user is most similar to U141954350? #######################
user <- "U141954350"
uIdx <- which(rownames(ratingData) == user)

simCosine <- similarity(x=ratingData[uIdx,], y=ratingData[-uIdx,],
                        method="cosine", which="users")
top10.Cosine <- colnames(simCosine)[order(simCosine, decreasing = T)][1:10]

# for Jaccard; split into 0/1 by below/above average
rdBin <- binarize(normalize(ratingData), minRating=0) 
simJaccard <- similarity(x=rdBin[uIdx, ], y=rdBin[-uIdx, ], 
                         method="jaccard", which="users")
top10.Jaccard <- colnames(simJaccard)[order(simJaccard, decreasing = T)][1:10]

top10 <- data.frame(Jaccard=top10.Jaccard, Cosine=top10.Cosine)
ExportTable(top10, "top10", sprintf("Top10 Users Similar to %s", user))

# Keep getting NAs, skip this for now.
# simPearson <- similarity(x=ratingData[uIdx,], y=ratingData[-uIdx,],
#                          method="pearson", which="users")
# colnames(simPearson[order(simPearson)])
