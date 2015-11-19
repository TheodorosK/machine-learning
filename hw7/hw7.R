rm(list=ls())

# Housekeeping ----------------------------------------------------------------

library(igraph)
library(igraphdata)

source("../utils/source_me.R", chdir=T)
CreateDefaultPlotOpts(WriteToFile = T)

###############################################################################
# ZACHARY'S KARATE CLUB #######################################################
###############################################################################

data(karate, package = "igraphdata")

# Vizualize -------------------------------------------------------------------

factions <- get.vertex.attribute(karate, "Faction")
leaders <- !grepl("Actor", V(karate)$name)

graph.color <- paste(gg_color_hue(2), "FF", sep="")[factions]
graph.color[!leaders] <- adjustcolor(graph.color[!leaders], alpha.f = 0.33)

karate.layout = layout.davidson.harel(karate)

PlotSetup('karate_network')
plot(karate, layout = karate.layout, 
     vertex.shape = c("square", "circle")[factions], 
     vertex.color = graph.color,
     vertex.size = 15,
     vertex.label.color = "#000000")
PlotDone()

# Split karate club into communities ------------------------------------------

PlotCommunities <- function(network, layout, factions, leaders, cl, fname) {
  
  if(is.hierarchical(cl))
    groups <- cutat(cl, 2)
  else
    groups <- membership(cl)
  
  graph.color <- paste(gg_color_hue(max(groups)), "FF", sep="")[groups]
  graph.color[!leaders] <- adjustcolor(graph.color[!leaders], alpha.f = 0.33)
  
  PlotSetup(fname)
  plot(network, layout = layout, 
       vertex.shape = c("square", "circle")[factions], 
       vertex.color = graph.color,
       vertex.size = 15,
       vertex.label.color = "#000000")
  PlotDone()
  
}

# http://stackoverflow.com/questions/2547402/standard-library-function-in-r-for-finding-the-mode
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# For each sub-group, the "true" faction is the modal faction
# Mis-classified if a vertex is not in the same faction as the rest of its group
# Assumes that each group is mostly right (so it works here...)
EvalCommunities <- function(network, cl, truecl) {
  gps <- groups(cl)
  nMissed <- 0
  #   missedVerts <- c()
  for(gp in gps) {
    gp.fac <- truecl[V(network)$name %in% gp] # factions of this sub-group
    nMissed <- nMissed + sum(gp.fac != Mode(gp.fac))
  #     missedVerts <- c(missedVerts, gp[gp.fac != Mode(gp.fac)])
  }
  #   return(list(nMissed, missedVerts))
  return(nMissed)
}

# Edge betweenness
# Note: does better when we tell it to ignore weights
cl <- cluster_edge_betweenness(karate, weights = NULL)
PlotCommunities(karate, karate.layout, factions, leaders, cl, 
                'edge_betweenness')
missed <- EvalCommunities(karate, cl, factions)
sprintf("%s algorithm mis-classified %d vertices", "edge_betweenness", missed)

# Fast greedy
cl = cluster_fast_greedy(karate)
PlotCommunities(karate, karate.layout, factions, leaders, cl, 
                'fast_greedy')
missed <- EvalCommunities(karate, cl, factions)
sprintf("%s algorithm mis-classified %d vertices", "fast_greedy", missed)

# Infomap
cl = cluster_infomap(karate)
PlotCommunities(karate, karate.layout, factions, leaders, cl, 
                'infomap')
missed <- EvalCommunities(karate, cl, factions)
sprintf("%s algorithm mis-classified %d vertices", "infomap", missed)

# Propagating labels
cl = cluster_label_prop(karate)
PlotCommunities(karate, karate.layout, factions, leaders, cl, 
                'label_prop')
missed <- EvalCommunities(karate, cl, factions)
sprintf("%s algorithm mis-classified %d vertices", "label_prop", missed)

# Leading eigenvector
# Warning message:
#   In cutat(cl, 2) : Cannot have that few communities
cl = cluster_leading_eigen(karate)
dg <- as.dendrogram(cl)
efac1 <- labels(dg[[1]])
efac2 <- labels(dg[[2]])
groups <- abs(as.numeric(V(karate)$name %in% efac1)-2)

# This alg gets 10 vertices wrong
sum(groups!=factions)

graph.color <- paste(gg_color_hue(max(groups)), "FF", sep="")[groups]
graph.color[!leaders] <- adjustcolor(graph.color[!leaders], alpha.f = 0.33)
PlotSetup("leading_eigen")
plot(karate, layout = karate.layout, 
     vertex.shape = c("square", "circle")[factions], 
     vertex.color = graph.color,
     vertex.size = 15,
     vertex.label.color = "#000000")
PlotDone()

# Multi-level optimization of modularity
cl = cluster_louvain(karate)
PlotCommunities(karate, karate.layout, factions, leaders, cl, 
                'louvain')
missed <- EvalCommunities(karate, cl, factions)
sprintf("%s algorithm mis-classified %d vertices", "louvain", missed)

# Optimal community structure
cl = cluster_optimal(karate)
PlotCommunities(karate, karate.layout, factions, leaders, cl, 
                'optimal')
missed <- EvalCommunities(karate, cl, factions)
sprintf("%s algorithm mis-classified %d vertices", "optimal", missed)

# Statistical meachanics
cl = cluster_spinglass(karate)
PlotCommunities(karate, karate.layout, factions, leaders, cl, 
                'spinglass')
missed <- EvalCommunities(karate, cl, factions)
sprintf("%s algorithm mis-classified %d vertices", "spinglass", missed)

# Short random walks
cl = cluster_walktrap(karate)
PlotCommunities(karate, karate.layout, factions, leaders, cl, 
                'walktrap')
missed <- EvalCommunities(karate, cl, factions)
sprintf("%s algorithm mis-classified %d vertices", "walktrap", missed)

# WIKIPEDIA ###################################################################
wiki <- LoadCacheTagOrRun("wikipedia", function() {
  require(curl)
  curl_download('https://github.com/ChicagoBoothML/MachineLearning_Fall2015/blob/master/Programming%20Scripts/Lecture08/hw/wikipedia.gml?raw=true',
                'wikipedia.gml', quiet=F, mode="wb")
  return(read.graph('wikipedia.gml', format='gml'))
})

# Vizualize -------------------------------------------------------------------
PlotCluster <- function(network, cl) {
  if(is.hierarchical(cl))
    groups <- cutat(cl, 2)
  else
    groups <- membership(cl)
  
  graph.color <- paste(gg_color_hue(max(groups)), "FF", sep="")[groups]
#   graph.color[!leaders] <- adjustcolor(graph.color[!leaders], alpha.f = 0.33)
  
#   PlotSetup(fname)
  plot(network, 
       vertex.color = graph.color,
       vertex.label.color = "#000000")
#        vertex.size = 15,
#   PlotDone()
}

# wiki.layout <- layout_nicely(wiki)
cl <- LoadCacheTagOrRun('wiki_infomap', function() cluster_infomap(wiki))

V(wiki)$membership <- membership(cl)
top10.idx <- order(table(membership(cl)), decreasing = T)[1:10]
wiki.sub <- induced.subgraph(wiki, membership(cl) %in% top10.idx)

member.table <- table(V(wiki.sub)$membership)
require(scales)
wiki.sub.palette <- alpha(gg_color_hue(length(member.table)), 0.5)
V(wiki.sub)$color <- wiki.sub.palette[
  sapply(V(wiki.sub)$membership, function(x) which(x == names(member.table)))]
V(wiki.sub)$label.color <- alpha(V(wiki.sub)$color, 1.0)

toKeep <- sample(length(V(wiki.sub)), 20)
V(wiki.sub)$label[-toKeep] <- ""
plot(wiki.sub, vertex.frame.color=NA, vertex.size=1, edge.arrow.mode='-')

# PlotCluster(wiki, cl)


