# RESET! ######################################################################
rm(list = ls())

require(jsonlite)
require(recommenderlab)

source("../utils/source_me.R", chdir=T)

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

rdNorm <- normalize(ratingData)
rdBin <- binarize(rdNorm, 0) # for Jaccard; split into 0/1 by below/above average

uIdx <- which(rownames(rdNorm) == "U141954350")

simCosine <- similarity(x = rdNorm[uIdx, ], y = rdNorm[-uIdx, ], 
                        method = "cosine", which = "users")
head(simCosine[order(simCosine, decreasing = T)])

simJaccard <- similarity(x = rdBin[uIdx, ], y = rdBin[-uIdx, ], 
                         method = "jaccard", which = "users")
simJaccard[order(simJaccard, decreasing = T)]


simPearson <- similarity(x = ratingData[uIdx, ], y = ratingData[-uIdx, ], 
                         method = "pearson", which = "users")
head(simPearson[order(simPearson)])

# Question 4: Recommend a video game to the user “U141954350” #################

rec <- Recommender(ratingData[-uIdx,], method = "POPULAR")

pre <- predict(rec, ratingData[uIdx,], n = 10)
pre <- as(pre, "list")[[1]]

preRat <- predict(rec, ratingData[uIdx,], type="ratings")
preRat <- as(preRat, "list")$U141954350

# How would the user rate the 10 most popular items?
preRat[names(preRat) %in% pre]

# What 10 items would get the highest rating?
preRat[order(preRat, decreasing = T)][1:10]
