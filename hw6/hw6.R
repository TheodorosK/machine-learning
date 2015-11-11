rm(list = ls())

require(jsonlite)
require(recommenderlab)

source("../utils/source_me.R", chdir=T)

# Load Data ###################################################################

fileConnection <- gzcon(file("videoGames.json.gz", "rb"))
dat <- LoadCacheTagOrRun('raw', stream_in, fileConnection)
close(fileConnection)
rm(fileConnection)

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

rdNorm <- normalize(ratingData)

uIdx <- which(rownames(rdNorm) == "U141954350")

simCosine <- order(similarity(x = rdNorm[uIdx, ], y = rdNorm[-uIdx, ], 
                              method = "cosine", which = "users"), 
                   decreasing = T)[1:10]

simPearson <- order(similarity(x = rdNorm[uIdx, ], y = rdNorm[-uIdx, ], 
                              method = "pearson", which = "users"), 
                   decreasing = T)[1:10]

simJaccard <- order(similarity(x = rdNorm[uIdx, ], y = rdNorm[-uIdx, ], 
                              method = "jaccard", which = "users"), 
                   decreasing = T)[1:10]

simDf <- data.frame(cosine = rownames(ratingData)[simCosine],
                    pearson = rownames(ratingData)[simPearson],
                    jaccard = rownames(ratingData)[simJaccard])

