rm(list=ls())

# Housekeeping ----------------------------------------------------------------

library(igraph)
library(igraphdata)

source("../utils/source_me.R", chdir=T)
CreateDefaultPlotOpts(WriteToFile = T)

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
cl <- cluster_edge_betweenness(karate)
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
PlotCommunities(karate, karate.layout, factions, leaders, cl, 
                'leading_eigen')
missed <- EvalCommunities(karate, cl, factions)
sprintf("%s algorithm mis-classified %d vertices", "leading_eigen", missed)

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
