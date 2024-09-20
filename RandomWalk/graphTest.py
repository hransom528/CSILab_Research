# Harris Ransom
# Random Walk Graph Class Testing
# 9/13/2024

# Imports
import numpy as np
import argparse
from Graph import Graph

# MAIN
if __name__ == "__main__":
	# Get input
	#numNodes = int(input("Enter number of nodes in graph: "))
	#numEdges = int(input("Enter number of edges in graph: "))

	# Create a graph
	#graph1 = Graph(numNodes, numEdges)

	# Test importCSV
	graph2 = Graph.importCSV("importCSV.csv")

	# Test getStationaryDistribution
	stationaryDist = graph2.getStationaryDistribution()

	# Test replaceRow
	replacementRow = [1,1,0,0,1]
	Graph.replaceRow(graph2, replacementRow, 0)

	# Test isDirected
	isDirected = graph2.isDirected()
	print(f"Imported graph directed? {isDirected}")

	# Test isDag
	isDag = graph2.isDag()
	print(f"Is imported graph a DAG? {isDag}")

	# Test exportCSV
	graph2.exportCSV("exportCSV.csv")

	# Output final graph (test __str__)
	print(graph2)
