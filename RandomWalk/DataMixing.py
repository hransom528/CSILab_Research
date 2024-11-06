# Harris Ransom
# Data Mixing - Glauber Dynamics

# Imports
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from math import exp
from Graph import Graph
from TVDistance import tvDistance
from graphGen import graphGen

# K(sigma, v) - Get number of different-classed neighbors for a given node
def getDifferentNeighbors(G, node):
    adjMatrix = G.A
    n = adjMatrix.shape[0]
    nodeType = G.getNodeType(node)

    # Test given node's neighbors
    diffNeighbors = 0
    for j in range(0, n):
        if (node == j):
            continue
        elif (adjMatrix[node, j] != 0) and (nodeType != G.getNodeType(j)):
            diffNeighbors += 1
    return diffNeighbors

# S(v) - Get the sum of the sigma-types of the neighbors of node v
def getNeighborTypeSum(G, node):
    neighborSet = G.getNeighborSet(node)
    neighborSum = 0
    for i in neighborSet:
        neighborSum += G.getNodeType(i)
    return neighborSum

# Gets "energy" of graph (sum of products of all node pairs)
def getEnergy(G):
    energy = 0
    for u in range(0, G.nodes):
        for v in range(0, G.nodes):
            energy += G.A[u, v]
    return (energy * 0.5)

# TODO: Gets the ratio/percentage of bad-to-good edges for data mixing
def getEdgeRatio(G):
    pass 

# TODO:
# 3.# of "Good" links vs. time

# Plots graph energy over time
def plotEnergy(times, energies):
    plt.figure()
    plt.plot(times, energies)
    plt.show()

# Plots histogram of percentage of different neighbors for each node
def plotDiffHist(G, bins=20):
    percentDiffNeighbors = []
    for i in range(0, G.nodes):
        row = np.asarray(G.A[i, :]).ravel().tolist()
        print(row)
        neighbors = G.getDegree(i)
        diffNeighbors = 0
        for j in range(0, len(row)):
            if (row[j] !=0) and (G.getNodeType(i) != G.getNodeType(j)):
                diffNeighbors += 1
        percentDiffNeighbors.append(diffNeighbors / neighbors)
    #print(percentDiffNeighbors)
    plt.figure()
    plt.hist(percentDiffNeighbors, bins)
    plt.show()
        
# Glauber Dynamics 
def GlauberDynamicsDataSwitch(G, times, temperature):
    energies = []
    for t in times:
        u = np.random.choice(np.arange(G.nodes))
        #diffNeighbors = G.getDifferentNeighborSet(u)
        #if (len(diffNeighbors) == 0):
        #    continue
        #v = np.random.choice(diffNeighbors)
        v = np.random.choice(G.getNeighborSet(u))

        if (G.getNodeType(u) != G.getNodeType(v)):
            # Calculates probability of switching
            exp1 = (G.getNodeType(u)*getNeighborTypeSum(G, v)) + (G.getNodeType(v)*getNeighborTypeSum(G, u)) 
            exp2 = (G.getNodeType(v)*getNeighborTypeSum(G, v)) + (G.getNodeType(u)*getNeighborTypeSum(G, u)) 
            exp3 = (G.getNodeType(v)*getNeighborTypeSum(G, u)) + (G.getNodeType(u)*getNeighborTypeSum(G, v)) 
            probSwitch = exp(-temperature * exp1) / (exp(-temperature * exp2) + exp(-temperature * exp3))
            print(f"Probability of switching: {probSwitch}")

            # Switches node type/data with probability probSwitch
            switch = np.random.choice([0, 1], p=[1-probSwitch, probSwitch])
            if (switch == 1):
                # Modify the graph's node types
                G.nodeTypes[u], G.nodeTypes[v] = G.nodeTypes[v], G.nodeTypes[u]
                
                # Modify the graph's adjacency matrix to reflect the node type switch   
                G.A[u, :] *= -1 
                G.A[v, :] *= -1

                # Plot/log graph state at each iteration
                plt.figure(f"figure{t+1}")
                G.plot_typed_graph(f"mixingPics/mixingGraph{t+1}.png")
                plt.close(f"figure{t+1}")

        # Log graph energy
        energies.append(getEnergy(G))
    return energies

# TODO: Metropolis-Hastings
def MetropolisHastingsDataSwitch(G, times, temperature):
    for t in times:
        v = np.random.choice(np.arange(G.nodes))
        # TODO: Check if v has different neighbor

# MAIN
if __name__ == "__main__":
    # Remove previous simulation pics (if any)
    files = glob.glob('mixingPics/*')
    for f in files:
        os.remove(f)

	# Create a typed graph object
    #G = Graph.importTypedCSV("graphData/mixingGraph.csv", [-1,1,1,-1,1,-1,-1,1])
    #G.plot_typed_graph("initialGraph.png")
    G2 = Graph.importTypedCSV("graphData/mixingGraph2.csv", [1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1])
    #G2.plot_typed_graph("mixingPics/mixingGraph0.png")
    G3 = graphGen(size=100, sparse_connections=10, plotGraph=False, path="graphData/mixingGraph3.csv")
    #G3.plot_typed_graph("mixingPics/mixingGraph0.png")

	# Generate time points
    n = 10000
    times = np.arange(n+1)

    # Test getDifferentNeighbors
    #print(getDifferentNeighbors(G2, 2))

    # Test getNeighborTypeSum
    #print(getNeighborTypeSum(G2, 5))

    # Run Glauber Dynamics data switching simulation
    print("Running Glauber Dynamics Algorithm...")
    energies = GlauberDynamicsDataSwitch(G3, times, 0.1) # TODO: What temperature to define?
    plotEnergy(times, energies)
    plotDiffHist(G3)
    # Observation: Temperature is inversely proportional to rate of switching

    # TODO: Run Metropolis-Hastings data switching simulation
    #print("Running Metropolis-Hastings Algorithm...")
    #MetropolisHastingsDataSwitch(G, times, 1)

    print("Finished Simulations!")