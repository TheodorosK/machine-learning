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

cl <- cluster_edge_betweenness(karate)
PlotCommunities(karate, karate.layout, factions, leaders, cl, 
                'edge_betweenness')

cl = cluster_fast_greedy(karate)
PlotCommunities(karate, karate.layout, factions, leaders, cl, 
                'fast_greedy')

cl = cluster_infomap(karate)
PlotCommunities(karate, karate.layout, factions, leaders, cl, 
                'infomap')

cl = cluster_label_prop(karate)
PlotCommunities(karate, karate.layout, factions, leaders, cl, 
                'label_prop')

# Warning message:
#   In cutat(cl, 2) : Cannot have that few communities
cl = cluster_leading_eigen(karate)
PlotCommunities(karate, karate.layout, factions, leaders, cl, 
                'leading_eigen')

cl = cluster_louvain(karate)
PlotCommunities(karate, karate.layout, factions, leaders, cl, 
                'louvain')

cl = cluster_optimal(karate)
PlotCommunities(karate, karate.layout, factions, leaders, cl, 
                'optimal')

cl = cluster_spinglass(karate)
PlotCommunities(karate, karate.layout, factions, leaders, cl, 
                'spinglass')

cl = cluster_walktrap(karate)
PlotCommunities(karate, karate.layout, factions, leaders, cl, 
                'walktrap')
