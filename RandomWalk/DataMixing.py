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

# No. of "Good" links (edges between nodes of different types)
def getGoodLinks(G):
    goodLinks = 0
    for u in range(0, G.nodes):
        for v in range(0, G.nodes):
            if (G.A[u, v] == 1) and (G.getNodeType(u) != G.getNodeType(v)):
                goodLinks += 1

    goodLinks = goodLinks // 2
    return goodLinks

# Gets the ratio/percentage of bad-to-good edges for data mixing
def getEdgeRatio(G):
    numGoodLinks = getGoodLinks(G)
    numBadLinks = G.edges - numGoodLinks
    return numBadLinks / numGoodLinks

# Plots graph energy over time
def plotEnergy(times, energies):
    plt.figure()
    plt.plot(times, energies)
    plt.title("Graph Energy Over Time")
    plt.xlabel("Time Steps")
    plt.ylabel("Energy")
    plt.show()

# Plots number of good links over time
def plotGoodLinks(times, numGoodLinks):
    plt.figure()
    plt.plot(times, numGoodLinks)
    plt.title("Number of Good Links Over Time")
    plt.xlabel("Time Steps")
    plt.ylabel("Number of Good Links")
    plt.show()

# Plots histogram of percentage of different neighbors for each node
def plotDiffHist(G, bins=20):
    percentDiffNeighbors = []
    for i in range(0, G.nodes):
        row = np.asarray(G.A[i, :]).ravel().tolist()
        neighbors = G.getDegree(i)
        diffNeighbors = 0
        for j in range(0, len(row)):
            if (row[j] !=0) and (G.getNodeType(i) != G.getNodeType(j)):
                diffNeighbors += 1
        percentDiffNeighbors.append(diffNeighbors / neighbors)
    #print(percentDiffNeighbors)
    plt.figure()
    plt.hist(percentDiffNeighbors, bins)
    plt.title("Histogram of Percentage of Different Neighbors")
    plt.xlabel("Percentage of Different Neighbors")
    plt.ylabel("Frequency")
    plt.show()
        
# Glauber Dynamics 
def GlauberDynamicsDataSwitch(G, times, temperature, plot=True):
    energies = []
    numGoodLinks = []
    for t in times:
        # Selects a random different-typed edge
        nodeList = np.unique(np.array(np.where(G.A == -1)[0]))
        u = np.random.choice(nodeList)
        neighborSet = G.getNeighborSet(u)
        v = np.random.choice(neighborSet)
        while (G.getNodeType(u) == G.getNodeType(v)):
            v = np.random.choice(neighborSet)

        #diffNeighbors = G.getDifferentNeighborSet(u)
        #if (len(diffNeighbors) == 0):
        #    continue
        #v = np.random.choice(diffNeighbors)
        
        # TODO: Update to choose from only good links
        '''
        while (G.getNodeType(u) == G.getNodeType(v)):
            while (len(neighborSet) > 0):
                i = np.random.choice(neighborSet)
                neighborSet.remove(i)
                if (G.getNodeType(i) != G.getNodeType(u)):
                    v = i
                    break
            u = np.random.choice(nodeList)
            neighborSet = G.getNeighborSet(u)'''

        if (G.getNodeType(u) != G.getNodeType(v)):
            # Calculates probability of switching
            exp1 = (G.getNodeType(u)*getNeighborTypeSum(G, v)) + (G.getNodeType(v)*getNeighborTypeSum(G, u)) 
            exp2 = (G.getNodeType(v)*getNeighborTypeSum(G, v)) + (G.getNodeType(u)*getNeighborTypeSum(G, u)) 
            exp3 = (G.getNodeType(v)*getNeighborTypeSum(G, u)) + (G.getNodeType(u)*getNeighborTypeSum(G, v)) 
            probSwitch = exp(-temperature * exp1) / (exp(-temperature * exp2) + exp(-temperature * exp3))
            print(f"{t}.) Probability of switching: {probSwitch}")

            # Switches node type/data with probability probSwitch
            switch = np.random.choice([0, 1], p=[1-probSwitch, probSwitch])
            if (switch == 1):
                # Modify the graph's node types
                G.nodeTypes[u], G.nodeTypes[v] = G.nodeTypes[v], G.nodeTypes[u]
                
                # Modify the graph's adjacency matrix to reflect the node type switch   
                G.A[u, :] *= -1 
                G.A[v, :] *= -1

                # Plot/log graph state at each iteration
                if (plot):
                    plt.figure(f"figure{t+1}")
                    G.plot_typed_graph(f"mixingPics/mixingGraph{t+1}.png")
                    plt.close(f"figure{t+1}")

        # Log graph energy
        energies.append(getEnergy(G))
        numGoodLinks.append(getGoodLinks(G))
    return energies, numGoodLinks

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
    print("Generating Graph...")
    #G = Graph.importTypedCSV("graphData/mixingGraph.csv", [-1,1,1,-1,1,-1,-1,1])
    #G.plot_typed_graph("initialGraph.png")
    #G2 = Graph.importTypedCSV("graphData/mixingGraph2.csv", [1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1])
    #G2.plot_typed_graph("mixingPics/mixingGraph0.png")
    #G3 = graphGen(size=100, sparse_connections=10, plotGraph=False, path="graphData/mixingGraph3.csv")
    #G3.plot_typed_graph("mixingPics/mixingGraph0.png")
    G4 = graphGen(size=1000, sparse_connections=50, p=0.1, plotGraph=False, path="graphData/mixingGraph4.csv")
    G = G4

	# Generate time points
    n = 200_000
    times = np.arange(n+1)

    # Test getDifferentNeighbors
    #print(getDifferentNeighbors(G2, 2))

    # Test getNeighborTypeSum
    #print(getNeighborTypeSum(G2, 5))

    # Run Glauber Dynamics data switching simulation
    print("Running Glauber Dynamics Algorithm...")
    energies, numGoodLinks = GlauberDynamicsDataSwitch(G, times, 0.05, plot=False) # TODO: What temperature to define?
    # (t=0.5, n = 5000 for 40 nodes)
    # (t=0.1, n = 50000 for 100 nodes)
    # (t=0.05, n=200000 for 1000 nodes)?

    plotEnergy(times, energies)
    plotGoodLinks(times, numGoodLinks)
    plotDiffHist(G)
    G.plot_typed_graph("finalGraph.png")
    # Observation: Temperature is inversely proportional to rate of switching

    # TODO: Run Metropolis-Hastings data switching simulation
    #print("Running Metropolis-Hastings Algorithm...")
    #MetropolisHastingsDataSwitch(G, times, 1)

    print("Finished Simulations!")