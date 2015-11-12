# RESET! ######################################################################
rm(list = ls())

require(jsonlite)
require(recommenderlab)
require(ggplot2)

source("../utils/source_me.R", chdir=T)
CreateDefaultPlotOpts(WriteToFile = T)

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

# Question 1: Which user has rated the most games? ############################

userMost <- which.max(rowCounts(ratingData))
rowCounts(ratingData[userMost, ])

table(rowCounts(ratingData))
max(scale(rowCounts(ratingData)))

PlotSetup("histo_users")
p <- qplot(rowCounts(ratingData), binwidth = 1) + 
  geom_vline(xintercept = rowCounts(ratingData[userMost, ]), color="red") + 
  labs(x = "Number of Ratings / User", y = "Frequency")
plot(p)
PlotDone()

# Question 2: Which game has been rated by the most users? ####################

gameMost <- which.max(colCounts(ratingData))
colCounts(ratingData[, gameMost])

table(colCounts(ratingData))
max(scale(colCounts(ratingData)))

PlotSetup("histo_games")
p <- qplot(colCounts(ratingData), binwidth = 3) + 
  geom_vline(xintercept = colCounts(ratingData[, gameMost]), color="red") + 
  labs(x = "Number of Unique Ratings / Game", y = "Frequency")
plot(p)
PlotDone()

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

# Question 4: Recommend a video game to the user “U141954350” #################

rec <- Recommender(ratingData, method = "POPULAR")

# Predict the top 10 games for the chosen user.
pre <- predict(rec, ratingData[uIdx,], type="topNList", n=10)
pre <- as(pre, "list")[[1]]
print(pre)

# Predict all game ratings for the chosen user.
preRat <- predict(rec, ratingData[uIdx,], type="ratings")
preRat <- as(preRat, "list")[[1]]
# Remove any items rated by the user.
rated <- as(ratingData[uIdx,], "list")[[1]]
preRat <- preRat[!(names(preRat) %in% names(rated))]
print(preRat)

# How would the user rate the 10 most popular items?
user.top10 <- sort(preRat[names(preRat) %in% pre], decreasing = T)

# What 10 items would get the highest rating?
user.highest <- sort(preRat, decreasing = T)[1:10]

ExportTable(as.data.frame(user.top10), "topn", 
            sprintf("%s's Predicted Top 10 from Recommender", user), 
            colnames = "Predicted Rating")
ExportTable(as.data.frame(user.highest), "highest",
            sprintf("%s's 10 Highest Rated Games", user),
            colnames = "Predicted Rating")
