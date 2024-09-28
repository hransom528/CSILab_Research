# Harris Ransom
# Random Walk Graph Class
# 9/10/2024

# Imports
import numpy as np
import pandas as pd

# Graph class
class Graph:
	# Constructor
	def __init__(self, numNodes=0, numEdges=0):
		#self.nodeList = []
		#self.edgeList = []
		self.nodes = numNodes
		self.edges = numEdges
		# TODO: Randomly assign edges in A
		self.A = np.zeros((numNodes, numNodes))

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
			if (i > 0):
				self.edges += 1
		self.edges = self.edges // 2
		# Convert the DataFrame to a numpy array
		self.A = df.to_numpy()
		return self

	# Export adjacency matrix to CSV
	def exportCSV(self, filename):
		pd.DataFrame(self.A).to_csv(filename, index=False, header=None)

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
		if (node < 1 or node > self.nodes):
			raise ValueError("Node out of bounds")
		return np.sum(self.A[node-1, :])

	# Getter method for set of node neighbors
	def getNeighborSet(self, node):
		neighbors = []
		index = node - 1
		row = self.A[index, :].getA1().tolist()
		for i in range(0, len(row)):
			if row[i]:
				neighbors.append(i+1)
		return neighbors

	# Gets the stationary distribution of the graph
	def getStationaryDistribution(self):
		# Get the row sums of the adjacency matrix
		degrees = np.sum(self.A, axis=1)

		edgeCount = np.sum(self.A, axis=0)

		# Normalize the row sums
		pi = degrees / (2.0 * self.edges)

		# Return the normalized row sums
		return pi
		
	# TODO: Graphically represent current graph
	# def plot_graph(self):
