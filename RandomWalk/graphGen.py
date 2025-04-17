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

# Helper function to get m-ary colormaps
'''
def get_cmap(n, name='hsv'):
    # Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    # RGB color; the keyword argument name must be a standard mpl colormap name.
    return plt.cm.get_cmap(name, n)'
'''

# Graph Generation Function
def graphGen(cluster_size=20, sparse_connections=3, p=0.5, path="graphData/generatedDisjointGraph.csv", plotGraph=False):
	# Create two disjoint complete graphs and join them
	G1 = nx.erdos_renyi_graph(cluster_size, p)
	G2 = nx.erdos_renyi_graph(cluster_size, p)
	G = nx.disjoint_union(G1, G2)

	# Sets binary cluster types
	node_types = [-1] * (2*cluster_size)
	color_map = ['blue'] * (2*cluster_size)
	for i in range(0, cluster_size):
		node_types[i] = 1
		color_map[i] = 'red'

	# Create sparse connections between clusters
	for i in range(sparse_connections):
		# Randomly select a node from the first cluster
		n1 = np.random.choice(np.arange(cluster_size))
		
		# Randomly select a node from the second cluster
		n2 = np.random.choice(np.arange(cluster_size)) + cluster_size

		# Add edge
		#print(f"({n1}, {n2})")
		G.add_edge(n1, n2)

	# Check for nodes with no connections
	for i in range(len(G.nodes)):
		if G.degree[i] == 0:
			node_type = node_types[i]
			if (node_type == 1):
				G.add_edge(i, np.random.choice(np.arange(cluster_size)))
			elif (node_type == -1):
				G.add_edge(i, np.random.choice(np.arange(cluster_size)) + cluster_size)

	# Output joined graph
	graphLayout = nx.spring_layout(G)
	if (plotGraph):
		nx.draw(G, pos=graphLayout, node_color=color_map, with_labels=False, node_size=40)
		plt.show()

	# Export adjacency matrix to CSV
	adjMatrix = nx.adjacency_matrix(G).toarray()
	for i in range(0, 2*cluster_size):
		for j in range(0, 2*cluster_size):
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

# Graph Generation for M-ary Erdos-Renyi clustered graph
def mAryGraphGen(m=3, cluster_size=20, sparse_connections=3, p=0.5, path="graphData/generatedClusteredGraph.csv", plotGraph=False):
	# Check m parameter
	if (m <= 1):
		raise ValueError(f"m is too small ({m}), needs to be at least 2!")
	elif (m == 2):
		G = graphGen(cluster_size, sparse_connections, p, path, plotGraph)
		return G

	# Create m disjoint clusters and join them
	G = nx.Graph()
	for i in range(m):
		Gi = nx.erdos_renyi_graph(cluster_size, p)
		G = nx.disjoint_union(G, Gi)
	
	# Set cluster types
	node_types = []
	for i in range(m):
		type_val = i+1
		cluster_types = [type_val] * cluster_size
		for t in cluster_types:
			node_types.append(t)

	# Create sparse connections between each combinational pair of clusters
	for i in range(m):
		for j in range(i, m):
			if (i != j):
				for k in range(sparse_connections):
					# Randomly select a node from the ith cluster
					n1 = np.random.choice(np.arange(i * cluster_size, (i*cluster_size)+cluster_size))

					# Randomly select a node form the jth cluster
					n2 = np.random.choice(np.arange(j * cluster_size, (j*cluster_size)+cluster_size))

					# Add edge
					#print(f"({n1}, {n2})")
					G.add_edge(n1, n2)

	# Check for nodes with no connections
	for i in range(len(G.nodes)):
		if G.degree[i] == 0:
			G.add_edge(i, np.random.choice(np.arange(G.nodes())))
	
	# Output joined graph
	graphLayout = nx.spring_layout(G)
	if (plotGraph):
		if (m > 9):
			raise ValueError("m-ary graph too big to plot (not enough colors)")
		else:
			# Set color map based on types
			colors = "rbgcmykw"
			color_map = []
			for i in range(m):
				cluster_color_type = colors[i]
				cluster_colors = [cluster_color_type] * cluster_size
				for c in cluster_colors:
					color_map.append(c)

			# Plot graph
			nx.draw(G, pos=graphLayout, node_color=color_map, with_labels=False, node_size=40)
			plt.show()

	# Export adjacency matrix to CSV
	adjMatrix = nx.adjacency_matrix(G).toarray()
	graph_df = pd.DataFrame(adjMatrix)
	graph_df.to_csv(path, header=False, index=False)

	# Create custom Graph object
	GraphObj = Graph.importTypedCSV(path, node_types, m=m)
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

# Generates a single complete graph with specified size
def completeGraphGen(size, path="graphData/generatedCompleteGraph.csv", plotGraph=False):
	G = nx.complete_graph(size)

	# Sets node types
	node_types = [-1] * (size)
	color_map = ['blue'] * (size)
	for i in range(0, size // 2):
		node_types[i] = 1
		color_map[i] = 'red'

	# Plot graph if configured
	graphLayout = nx.spring_layout(G)
	if (plotGraph):
		nx.draw(G, pos=graphLayout, node_color=color_map, with_labels=False, node_size=40)
		plt.show()

	# Export adjacency matrix to CSV
	adjMatrix = nx.adjacency_matrix(G).toarray()
	for i in range(0, size):
		for j in range(0, size):
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

# Generates a single complete graph with specified size
def erdosRenyiGraphGen(size, p, path="graphData/generatedErdosRenyiGraph.csv", plotGraph=False):
	G = nx.erdos_renyi_graph(size, p)

	# Sets node types
	node_types = [-1] * (size)
	color_map = ['blue'] * (size)
	for i in range(0, size // 2):
		node_types[i] = 1
		color_map[i] = 'red'

	# Plot graph if configured
	graphLayout = nx.spring_layout(G)
	if (plotGraph):
		nx.draw(G, pos=graphLayout, node_color=color_map, with_labels=False, node_size=40)
		plt.show()

	# Export adjacency matrix to CSV
	adjMatrix = nx.adjacency_matrix(G).toarray()
	for i in range(0, size):
		for j in range(0, size):
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

# Generates a single d-regular graph with specified size
def dRegularGraphGen(size, d, path="graphData/generatedRegularGraph.csv", plotGraph=False):
	G = nx.random_regular_graph(d, size)

	# Sets node types
	node_types = [-1] * (size)
	color_map = ['blue'] * (size)
	for i in range(0, size // 2):
		node_types[i] = 1
		color_map[i] = 'red'

	# Plot graph if configured
	graphLayout = nx.spring_layout(G)
	if (plotGraph):
		nx.draw(G, pos=graphLayout, node_color=color_map, with_labels=False, node_size=40)
		plt.show()

	# Export adjacency matrix to CSV
	adjMatrix = nx.adjacency_matrix(G).toarray()
	for i in range(0, size):
		for j in range(0, size):
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
