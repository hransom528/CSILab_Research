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

# Graph Generation Function
def graphGen(size=20, sparse_connections=3, p=0.5, path="graphData/generatedDisjointGraph.csv", plotGraph=False):
	# Create two disjoint complete graphs and join them
	G1 = nx.erdos_renyi_graph(size, p)
	G2 = nx.erdos_renyi_graph(size, p)
	G = nx.disjoint_union(G1, G2)

	# Sets binary cluster types
	node_types = [-1] * (2*size)
	color_map = ['blue'] * (2*size)
	for i in range(0, size):
		node_types[i] = 1
		color_map[i] = 'red'

	# Create sparse connections between clusters
	for i in range(sparse_connections):
		# Randomly select a node from the first cluster
		n1 = np.random.choice(np.arange(size))
		
		# Randomly select a node from the second cluster
		n2 = np.random.choice(np.arange(size)) + size

		# Add edge
		#print(f"({n1}, {n2})")
		G.add_edge(n1, n2)

	# Output joined graph
	graphLayout = nx.spring_layout(G)
	if (plotGraph):
		nx.draw(G, pos=graphLayout, node_color=color_map, with_labels=False, node_size=40)
		plt.show()

	# Export adjacency matrix to CSV
	adjMatrix = nx.adjacency_matrix(G).toarray()
	for i in range(0, 2*size):
		for j in range(0, 2*size):
			if (adjMatrix[i, j] != 0):
				adjMatrix[i, j] = (node_types[i] * node_types[j])
	graph_df = pd.DataFrame(adjMatrix)
	graph_df.to_csv(path, header=False, index=False)

	# Create custom Graph object
	GraphObj = Graph.importTypedCSV(path, node_types)
	GraphObj.layout = graphLayout
	GraphObj.edges = G.number_of_edges()

	# Export node types to file
	with open(path+".types", "w") as output:
		outStr = str(node_types)
		outStr = outStr.replace("[", "")
		outStr = outStr.replace("]", "")
		output.write(outStr)

	# Return custom graph object
	return GraphObj

# MAIN
if __name__ == "__main__":
	# Handles command-line arguments
	parser = argparse.ArgumentParser(
                    prog='GraphGen',
                    description='Generates complete graphs for data mixing experiments')
	parser.add_argument("size", default=20, help="Size of an individual cluster")
	parser.add_argument("connections", default=3, help="No. of sparse connections")
	parser.add_argument("-p", "--plot", default=False, help="Plot generated graph", action="store_true")
	parser.add_argument("-o", "--out", default="graphData/generatedDisjointGraph.csv", help = "Output path")

	# Get input variables
	args = parser.parse_args()
	size = int(args.size)
	connections = int(args.connections)
	toPlotGraph = args.plot	

	# Generate graph based on parameters
	if (args.out):
		print(args.out)
		G = graphGen(size, connections, plotGraph=toPlotGraph, path=args.out)
	else:
		G = graphGen(size, connections, plotGraph=toPlotGraph)
	#G.plot_typed_graph()
