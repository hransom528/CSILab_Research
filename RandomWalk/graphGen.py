# Haris Ransom
# Clustered Graph Generation Script
# 10/18/2024

# Imports
import argparse
import networkx as nx
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from Graph import Graph
from DataMixing import GlauberDynamicsDataSwitch

# Graph Generation Function
def graphGen(size, sparse_connections, path="graphData/generatedDisjointGraph.csv", plotGraph=False):
	# Create two disjoint complete graphs and join them
	G1 = nx.complete_graph(size)
	G2 = nx.complete_graph(size)
	G = nx.disjoint_union(G1, G2)

	# Sets binary cluster types
	node_types = [-1] * (2*size)
	color_map = ['blue'] * (2*size)
	for i in range(0, size):
		node_types[i] = 1
		color_map[i] = 'red'

	# Create sparse connections between clusters
	for i in range(connections):
		# Randomly select a node from the first cluster
		n1 = np.random.choice(np.arange(size))
		
		# Randomly select a node from the second cluster
		n2 = np.random.choice(np.arange(size)) + size

		# Add edge
		#print(f"({n1}, {n2})")
		G.add_edge(n1, n2)

	# Output joined graph
	#print(color_map)
	#print(G.nodes)
	if (plotGraph):
		nx.draw(G, node_color=color_map, with_labels=True)
		plt.show()

	# Export adjacency matrix to CSV
	adjMatrix = nx.adjacency_matrix(G).toarray()
	for i in range(0, 2*size):
		for j in range(0, 2*size):
			if (adjMatrix[i, j] != 0):
				adjMatrix[i, j] = node_types[j]
	graph_df = pd.DataFrame(adjMatrix)
	graph_df.to_csv("graphData/generatedDisjointGraph.csv", header=False, index=False)

	# Create custom Graph object
	GraphObj = Graph.importTypedCSV("graphData/generatedDisjointGraph.csv", node_types)
	return GraphObj

# MAIN
if __name__ == "__main__":
	#parser = argparse.ArgumentParser()

	size = 20
	connections = 3
	G = graphGen(size, connections, plotGraph=False)
	G.plot_typed_graph()
