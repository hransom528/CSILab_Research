# Harris Ransom
# Random Walk Graph Class
# 9/10/2024

# Imports
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Graph class
class Graph:
	# Constructor
	def __init__(self, numNodes=0, numEdges=0, isTyped=False, nodeTypes=[], m=2):
		# Basic graph fields
		#self.nodeList = []
		#self.edgeList = []
		self.nodes = numNodes
		self.edges = numEdges # TODO: Randomly assign edges in A
		self.A = np.zeros((numNodes, numNodes))

		# Random Walk Fields
		self.currentNode = 0
		self.nodesVisited = []

		# Typed Graph Fields
		self.typed = isTyped
		if (isTyped):
			if (len(nodeTypes) == numNodes):
				self.nodeTypes = nodeTypes
				self.nodeDists = []
				self.m = m
				for i in range(len(nodeTypes)):
					self.nodeDists[i] = self.calcNeighborhoodDist(i)
			else:
				raise ValueError("nodeTypes does not fit the dimensions of graph")
		
		# Plotting fields
		self.layout = {}

	# String constructor
	def __str__(self):
		return np.array2string(self.A)

	# CSV import function
	@staticmethod
	def importCSV(filename):
		# Read the CSV file
		df = pd.read_csv(filename, header=None)

		# Create new graph object
		self = Graph()

		# Get the number of nodes
		if (df.shape[0] != df.shape[1]):
			raise ValueError("Adjacency matrix not square!")
		self.nodes = df.shape[0]

		# Initialize adjacency matrix from dataframe
		self.A = df.to_numpy()

		# Get the number of edges
		self.edges = 0
		for i in np.nditer(self.A):
			if (i != 0):
				self.edges += 1
		self.edges = self.edges // 2

		# Misc. fields
		self.typed = False

		# Convert the DataFrame to a numpy array
		self.A = df.to_numpy()
		return self

	# Typed graph CSV import
	@staticmethod
	def importTypedCSV(filename, nodeTypes, m=2):
		self = Graph.importCSV(filename)
		if (len(nodeTypes) != self.nodes):
			raise ValueError("nodeTypes does not fit the dimensions of graph")
		else:
			self.nodeTypes = nodeTypes
			self.typed = True
			self.nodeDists = []
			self.m = m
			for i in range(len(nodeTypes)):
				self.nodeDists.append(self.calcNeighborhoodDist(i))
		return self

	# Export adjacency matrix to CSV
	def exportCSV(self, filename):
		pd.DataFrame(self.A).to_csv(filename, index=False, header=None)
		if (self.typed):
			with open(f'{filename}.types', 'w') as outfile:
				outfile.write('\n'.join(str(i) for i in self.nodeTypes))

	# Inserts a pre-defined row into the adjacency matrix
	def insertRow(self, row, index):
		# Check input row
		if type(row) is list:
			row = np.asarray(row)
		if (row.size != self.nodes):
			raise ValueError("Row size does not match number of nodes")
		if (row.shape == (1, self.nodes)):
			row = row.T
		
		# Check input index
		if (index < 0 or index >= self.nodes):
			raise ValueError("Index out of bounds")
		
		# Insert the row into the adjacency matrix
		self.A = np.insert(self.A, index, row, axis=1)
		return self.A

	# Replaces a row in the adjacency matrix at index with new row data
	def replaceRow(self, row, index):
		# Check input row
		if type(row) is list:
			row = np.asarray(row)
		if (row.size != self.nodes):
			raise ValueError("Row size does not match number of nodes")
		if (row.shape == (1, self.nodes)):
			row = row.T

		# Check input index
		if (index < 0 or index >= self.nodes):
			raise ValueError("Index out of bounds")
		
		# Replace row of adjacency matrix
		self.A[index, :] = row
		return self.A

	# Check if graph is directed or undirected
	def isDirected(self):
		a, b = self.A.shape
		if a == b and (np.transpose(self.A) == self.A).all(): # Symmetric matrix (undirected)
			return False
		return True
	
	# Check if graph is a Directed Acyclic Graph (DAG)
	def isDag(self):
		# Check if graph is directed
		isDirected = self.isDirected()
		if (not isDirected):
			return False

		# Check diagonals for cycles
		diagonals = np.array(self.A.diagonal()).flatten()
		for i in diagonals:
			if (i == 1):
				return False
			
		return True

	# Get the degree of a node
	def getDegree(self, node):
		if (node < 0 or node >= self.nodes):
			raise ValueError("Node out of bounds")
		return np.sum(abs(self.A[node, :]))

	# Getter method for set of node neighbors
	def getNeighborSet(self, node):
		neighbors = []
		row = np.asarray(self.A[node, :]).ravel().tolist()
		for i in range(0, len(row)):
			if row[i]:
				neighbors.append(i)
		return neighbors

	# Returns node value (type)
	def getNodeType(self, node):
		if (self.typed):
			if (node < 0) or (node > self.nodes):
				raise ValueError("Node value out of bounds")
			return self.nodeTypes[node]
		else:
			raise ValueError("Graph is not typed")

	# Gets set of node neighbors that are of a different type
	def getDifferentNeighborSet(self, node):
		if (not self.typed):
			raise ValueError("Graph is not typed")
		
		nodeType = self.getNodeType(node)
		neighbors = self.getNeighborSet(node)
		diffNeighbors = []
		for i in neighbors:
			if (self.getNodeType(i) != nodeType):
				diffNeighbors.append(i)
		return diffNeighbors

	# Gets the stationary distribution of the graph
	# Note: This is for a simple random walk, not M.H.
	def getStationaryDistribution(self):
		# Get the row sums of the adjacency matrix
		degrees = np.sum(self.A, axis=1)

		#edgeCount = np.sum(self.A, axis=0)

		# Normalize the row sums
		pi = degrees / (2.0 * self.edges)

		# Return the normalized row sums
		return pi

	# Graphically represent current graph	
	def plot_graph(self, path=""):
		Graphtype = nx.Graph()
		G = nx.from_numpy_array(self.A)
		nx.draw_networkx(G)
		if (path != ""):
			plt.savefig(path)
		else:
			plt.show()
	
	# Graphically represent current typed graph
	# TODO: Update with M-ary plotting
	def plot_typed_graph(self, path="", m=2):
		# Check if graph is typed
		if (not self.typed):
			raise ValueError("Graph is not typed")
		
		# Create networkx graph
		Graphtype = nx.Graph()
		G = nx.from_numpy_array(self.A, create_using=nx.Graph)

		# Assign node types to graph
		colors = "rbgcmykw"
		color_map = []
		for i in range(0, self.nodes):
			G.nodes[i]['type'] = self.nodeTypes[i]
			if (m == 2):
				if (self.nodeTypes[i] == -1):
					color_map.append('blue')
				else:
					color_map.append('red')
			elif (m <= 8):
				color_map.append(colors[self.nodeTypes[i]])
			else:
				raise ValueError("m cannot exceed 8 for plotting!")

		# Draw the graph
		#print(G.nodes.data())
		if (not self.layout):
			self.layout = nx.spring_layout(G)
		nx.draw(G, pos=self.layout, node_color=color_map, node_size=160, with_labels=False, width=1)
		if (path != ""):
			plt.savefig(path)
			plt.close()
		else:
			plt.show()

	# Gets transistion matrix based on graph and stationary distribution
	def getTransistionMatrix(self, pi):
		# 1.) Get an initial proposed transition matrix
		n = self.nodes
		Q = np.zeros((n, n))
		for i in range(0, n):
			for j in range(0, n):
				if (self.A[i, j] > 0):
					Q[i, j] = 1 / self.getDegree(i+1)
				else:
					Q[i, j] = 0

		# 2.) Check Q for probability distribution criteria
		for i in range(0, n):
			if (np.sum(Q[i, :]) != 1):
				raise ValueError(f"Q[{i}, :] is not a probability distribution")
		#print(Q)

		# 3.) Calculates P from Q and pi
		P = np.zeros((n, n))
		for i in range(0, n):
			for j in range(0, n):
				if (i != j):
					if (Q[i, j] > 0) and (Q[j, i] > 0):
						P[i, j] = Q[i, j] * min(1, (pi[j] * Q[j, i] / pi[i] * Q[i, j]))
					else:
						P[i, j] = 0
		for i in range(0, n):
			P[i, i] = 1 - np.sum(P[i, :])
		
		return P
	
	# Calculates the neighboorhood distribution for a given node
	def calcNeighborhoodDist(self, node):
		# Get row of node from adjacency matrix
		row = self.A[node, :]

		# Get the type of the current node
		baseType = self.nodeTypes[node]
		
		# Get number of neighbors and neighbors that are different
		neighborCounts = [0] * self.m
		totalNeighbors = 0
		diffNeighbors = 0
		for i in range(len(row)):
			# Don't count current node
			if (i == node):
				continue

			# If a neighbor exists
			if (row[i] != 0):
				totalNeighbors += 1
				if (self.nodeTypes[i] != baseType):
					diffNeighbors += 1

				if (self.nodeTypes[i] == -1): # binary case (m=2) TODO: Fix this
					neighborCounts[1] += 1
				else:
					neighborCounts[self.nodeTypes[i] - 1] += 1

		# Calculate and return percent different neighbors
		if (totalNeighbors <= 0): # No neighbors
			return 0
		percentDiffNeighbors = diffNeighbors / float(totalNeighbors)
		neighborDist = [x / float(totalNeighbors) for x in neighborCounts]
		return neighborDist
