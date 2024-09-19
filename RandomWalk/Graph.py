# Harris Ransom
# Random Walk Graph Class
# 9/10/2024

# Imports
import numpy as np
import pandas as pd
import networkx as nx

# Graph class
class Graph:
	# Constructor
	def __init__(self, numNodes=0, numEdges=0):
		#self.nodeList = []
		#self.edgeList = []
		self.nodes = numNodes
		self.edges = numEdges
		self.A = np.zeros((numNodes, numNodes))

	# TODO: Create CSV import function
	def importCSV(self, filename):
		# Read the CSV file
		df = pd.read_csv(filename)

		# Get the number of nodes
		if (df.shape[0] != df.shape[1]):
			raise ValueError("Adjacency matrix not square!")
		self.nodes = df.shape[0]
		
		# Convert the DataFrame to a numpy array
		self.A = df.to_numpy()

	# Export adjacency matrix to CSV
	def exportCSV(self, filename):
		pd.DataFrame(self.A).to_csv(filename, index=False)

	# Getter method for set of node neighbors
	def getNeighborSet(self, node):
		neighbors = []
		index = node - 1
		row = self.A[index, :].getA1().tolist()
		for i in range(0, len(row)):
			if row[i]:
				neighbors.append(i+1)
		return neighbors

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
		
	# Replaces a row in the adjacency matrix at index with new row data
	def replaceRow(self, row, index):
		pass


	# Gets the stationary distribution of the graph
	def getStationaryDistribution(self):
		# Get the row sums of the adjacency matrix
		degrees = np.sum(self.A, axis=1)

		# Normalize the row sums
		pi = degrees / (2.0 * self.edges)

		# Return the normalized row sums
		return pi

	# TODO: Graphically represent current graph
	# def plot_graph(self):
