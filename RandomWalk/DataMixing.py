# Harris Ransom
# Data Mixing - Glauber Dynamics

# Imports
import numpy as np
import matplotlib.pyplot as plt
from math import exp
from Graph import Graph
from TVDistance import tvDistance

# K(sigma, v) - Get number of different-classed neighbors for a given node
def getDifferentNeighbors(G, sigma, node):
    adjMatrix = G.A
    n = adjMatrix.shape[0]
    nodeType = G.getNodeType(node)

    # Test given node's neighbors
    diffNeighbors = 0
    for j in range(0, n):
        if (node == j):
            continue
        elif (nodeType != G.getNodeType(j)):
            diffNeighbors += 1
    return diffNeighbors

# S(v) - Get the sum of the sigma-types of the neighbors of node v
def getNeighborTypeSum(G, node):
    neighborSet = G.getNeighborSet(node)
    neighborSum = 0
    for i in neighborSet:
        neighborSum += i
    return neighborSum

# Glauber Dynamics 
def GlauberDynamicsDataSwitch(G, times, temperature):
    for t in times:
        u = np.random.choice(np.arange(G.nodes))
        v = np.random.choice(np.arange(G.nodes))
        if (G.getNodeType(u) != G.getNodeType(v)):
            # Calculates probability of switching
            exp1 = (G.getNodeType(u)*getNeighborTypeSum(G, v)) + (G.getNodeType(v)*getNeighborTypeSum(G, u)) 
            exp2 = (G.getNodeType(v)*getNeighborTypeSum(G, v)) + (G.getNodeType(u)*getNeighborTypeSum(G, u)) 
            # TODO: Calc exp3 as probability of configuration after exchanging the data
            probSwitch = exp(-temperature * exp1) / (exp(-temperature * exp2) + exp(-temperature * exp1))

            # TODO: Switches node type/data with probability probSwitch
            switch = np.random.choice([0, 1], p=[1-probSwitch, probSwitch])
            if (switch == 1):
                G.nodeTypes[u], G.nodeTypes[v] = G.nodeTypes[v], G.nodeTypes[u]

        # TODO: Plot/log graph state at each iteration
        G.plot_graph()

# TODO: Metropolis-Hastings
def MetropolisHastingsDataSwitch(G, times, temperature):
    for t in times:
        v = np.random.choice(np.arange(G.nodes))
        # TODO: Check if v has different neighbor

# MAIN
if __name__ == "__main__":
	# Create a typed graph object
    #G = Graph.importCSV("mixingGraph.csv")
    G = Graph.importTypedCSV("mixingGraph.csv", [-1,1,1,-1,1,-1,-1,1])
    print(G)
    G.plot_graph()
    
	# Generate time points
    n = 100
    times = np.arange(n+1)

    # Test getDifferentNeighbors
    getDifferentNeighbors(G, [0.1, 0.5, 0.4], 5)

    # Run Glauber Dynamics data switching simulation
    GlauberDynamicsDataSwitch(G, times, 1) # TODO: What temperature to define?

    # TODO: Run Metropolis-Hastings data switching simulation