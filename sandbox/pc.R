library(pscl)

votes <- read.csv("../data/rollcall-votes.csv")
legis <- read.csv("../data/rollcall-members.csv")

pcavote <- prcomp(votes, scale=TRUE)
plot(pcavote, main="")
mtext(side=1, "Rollcall-Vote Principle Components",  line=1, font=2)
