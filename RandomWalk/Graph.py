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
		self.nodes = numNodes
		self.edges = numEdges
		self.A = np.zeros((numNodes, numNodes))

	# TODO: Create CSV import function

	# TODO: Implement getter method for set of node neighbors
	#def getNeighborSet(self, node):

	# TODO: Implement inserting a pre-defined row into the adjacency matrix
	#def insert_row(self, row, index):

	# TODO: Gets the stationary distribution of the graph
	def getStationaryDistribution(self):
		# Get the row sums of the adjacency matrix
		degrees = np.sum(self.A, axis=1)

		# Normalize the row sums
		pi = degrees / (2.0 * self.edges)

		# Return the normalized row sums
		return pi

	# TODO: Graphically represent current graph
	# def plot_graph(self):

