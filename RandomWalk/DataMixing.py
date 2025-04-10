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
from graphGen import graphGen, mAryGraphGen

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
    energy = np.sum(G.A)
    #for u in range(0, G.nodes):
    #    for v in range(0, G.nodes):
    #        energy += G.A[u, v]
    return (energy * 0.5)

# Get the energy of an M-ary graph
def mAryGetEnergy(m, G):
    ideal = 1 - (1 / float(m))
    idealDist = [ideal] * G.nodes
    energy = tvDistance(G.nodeDists, idealDist)
    return energy

# No. of "Good" links (edges between nodes of different types)
def getGoodLinks(G):
    goodLinks = 0
    for u in range(0, G.nodes):
        for v in range(0, G.nodes):
            if (G.A[u, v] != 0) and (G.getNodeType(u) != G.getNodeType(v)):
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
def plotGoodLinks(G, times, numGoodLinks):
    numGoodLinks = np.array(numGoodLinks)
    print(f"Total edges: {G.edges}")
    print(f"Total good links: {np.max(numGoodLinks)}")
    plt.figure()
    plt.plot(times, numGoodLinks / G.edges)
    plt.title("Percent of Good Links Over Time")
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
            if (row[j] != 0) and (G.getNodeType(i) != G.getNodeType(j)):
                diffNeighbors += 1
        percentDiffNeighbors.append(diffNeighbors / neighbors)
    #print(percentDiffNeighbors)
    plt.figure()
    plt.hist(percentDiffNeighbors, bins)
    plt.xlim(xmin=-0.05, xmax = 1.000)
    plt.title("Histogram of Percentage of Different Neighbors")
    plt.xlabel("Percentage of Different Neighbors")
    plt.ylabel("Frequency")
    plt.show()
        
# Glauber Dynamics 
def GlauberDynamicsDataSwitch(G, times, temperature, plot=True, samplingSize=100):
    energies = []
    numGoodLinks = []
    sampleTimes = []
    nodeList = np.unique(np.array(np.where(G.A == -1)[0]))
    for t in times:
        # Selects a random different-typed edge
        u = np.random.choice(nodeList)
        neighborSet = G.getDifferentNeighborSet(u)
        v = np.random.choice(neighborSet)
        while (G.getNodeType(u) == G.getNodeType(v)):
            v = np.random.choice(neighborSet)
        
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
            
            if (t % samplingSize == 0):
                print(f"{t}.) Probability of switching: {probSwitch}")

            # Switches node type/data with probability probSwitch
            switch = np.random.choice([0, 1], p=[1-probSwitch, probSwitch])
            if (switch == 1):
                # Modify the graph's node types
                G.nodeTypes[u], G.nodeTypes[v] = G.nodeTypes[v], G.nodeTypes[u]
                
                # Modify the graph's adjacency matrix to reflect the node type switch   
                G.A[u, :] *= -1 
                G.A[v, :] *= -1

                # Edit nodelist to reflect changes
                for i in range(len(G.A[u, :])):
                    # TODO: Fix nodeList depleting itself
                    if (i in nodeList) and (G.A[u, i] != -1) and (G.A[v, i] != -1):
                        ind = np.where(nodeList == i)
                        nodeList = np.delete(nodeList, ind)

                    # For row u
                    if ((not (i in nodeList)) and (G.A[u, i] == -1)):
                        nodeList = np.append(nodeList, i)
                    
                    # For row v
                    if ((not (i in nodeList)) and (G.A[v, i] == -1)):
                        nodeList = np.append(nodeList, i)

                # Plot/log graph state at each iteration
                if (plot):
                    plt.figure(f"figure{t+1}")
                    G.plot_typed_graph(f"mixingPics/mixingGraph{t+1}.png")
                    plt.close(f"figure{t+1}")

        # Log graph energy every xth sample
        if (t % samplingSize == 0):
            sampleTimes.append(t)
            energies.append(getEnergy(G))
            numGoodLinks.append(getGoodLinks(G))
        if (t == len(times)//4):
            plt.figure("middleGraph")
            G.plot_typed_graph("mixingPics/middleGraph.png")
            plt.close("middleGraph")
    return sampleTimes, energies, numGoodLinks, G

# Calculates the probability of switching nodes in an M-ary graph
def mAryProbSwitch(m, G, u, v, temperature):
    idealVal = 1 - (1/float(m)) 
    uType = G.nodeTypes[u]
    vType = G.nodeTypes[v]
    '''
    uDist = G.nodeDists[u]
    vDist = G.nodeDists[v]
    tvU = uDist - ideal
    tvV = vDist - ideal
    '''
    # Get all nodes in the system
    uNeighbors = G.getNeighborSet(u)
    vNeighbors = G.getNeighborSet(v)
    neighborNodes = list(set(uNeighbors + vNeighbors))

    # Get initial percent different neighobrs for each node in the system
    diffNeighborDists = [G.nodeDists[n] for n in neighborNodes]
    idealDist = [idealVal] * len(neighborNodes)
    priorDiff = tvDistance(diffNeighborDists, idealDist)

    # Get posterior percent different neighbors, assuming the switch occurs
    G.nodeTypes[u] = vType
    G.nodeTypes[v] = uType
    posteriorNeighborDists = []
    for n in neighborNodes:
        posteriorNeighborDists.append(G.calcNeighborhoodDist(n))
    posteriorDiff = tvDistance(posteriorNeighborDists, idealDist)

    # Undo the assumed node switch
    G.nodeTypes[u] = uType
    G.nodeTypes[v] = vType

    # Calculate switching probability based off of change in TV distance
    totalDiff = priorDiff - posteriorDiff 
    #print(priorDiff)
    #print(posteriorDiff)
    #print(totalDiff)
    #return
    switchProb = (1 / float(1 + exp(-temperature * totalDiff)))
    return switchProb


# Glauber Dynamics M-ary Data Switching
def mAryGlauberDynamicsDataSwitch(m, G, times, temperature, plot=False, samplingSize=100):
    energies = []
    numGoodLinks = []
    sampleTimes = []
    nodeList = np.arange(G.nodes)

    for t in times:
        # Select two random nodes of different types
        neighborSet = []
        while (len(neighborSet) == 0):
            u = np.random.choice(nodeList)
            neighborSet = G.getDifferentNeighborSet(u)
        v = np.random.choice(neighborSet)
        while (G.getNodeType(u) == G.getNodeType(v)):
            v = np.random.choice(neighborSet)

        # Perform mixing if nodes are different
        if (G.getNodeType(u) != G.getNodeType(v)):
            # Calculates probability of switching
            probSwitch = mAryProbSwitch(m, G, u, v, temperature)

            # Print switching probability during a sample time
            if (t % samplingSize == 0):
                print(f"{t}.) Probability of switching: {probSwitch}")

            # Switches node type/data with probability probSwitch
            switch = np.random.choice([0, 1], p=[1-probSwitch, probSwitch])
            if (switch == 1):
                # Performs switch
                uType = G.nodeTypes[u]
                vType = G.nodeTypes[v]
                G.nodeTypes[u] = vType
                G.nodeTypes[v] = uType

                # Recalculate node dists
                # TODO: Make this smart (i.e. don't do it for all the nodes)
                for i in range(G.nodes):
                    G.nodeDists[i] = G.calcNeighborhoodDist(i)

                # Plot/log graph state at each iteration
                if (plot):
                    plt.figure(f"figure{t+1}")
                    G.plot_typed_graph(f"mixingPics/mixingGraph{t+1}.png")
                    plt.close(f"figure{t+1}")

        # Log graph energy every xth sample
        if (t % samplingSize == 0):
            sampleTimes.append(t)
            energies.append(mAryGetEnergy(m, G))
            numGoodLinks.append(getGoodLinks(G))

        # Plot approximate "middle" of mixing
        if (plot and (t == len(times)//4)):
            plt.figure("middleGraph")
            G.plot_typed_graph("mixingPics/middleGraph.png")
            plt.close("middleGraph")
    return sampleTimes, energies, numGoodLinks, G

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

    '''
    # --- Part 1: Binary mixing ---
	# Create a typed graph object
    print("Generating Graph...")
    #G = Graph.importTypedCSV("graphData/mixingGraph.csv", [-1,1,1,-1,1,-1,-1,1])
    #G.plot_typed_graph("initialGraph.png")
    #G2 = Graph.importTypedCSV("graphData/mixingGraph2.csv", [1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1])
    #G2.plot_typed_graph("mixingPics/mixingGraph0.png")
    G3 = graphGen(size=100, sparse_connections=10, plotGraph=False, path="graphData/mixingGraph3.csv")
    #G3.plot_typed_graph("mixingPics/mixingGraph0.png")
    #G4 = graphGen(size=1000, sparse_connections=50, p=0.1, plotGraph=False, path="graphData/mixingGraph4.csv")
    G = G3

    # Create initial figures
    G.plot_typed_graph("mixingPics/startGraph.png")
    plotDiffHist(G, bins=5)

	# Generate time points
    n = 100_000
    times = np.arange(n+1)

    # Test getDifferentNeighbors
    #print(getDifferentNeighbors(G2, 2))

    # Test getNeighborTypeSum
    #print(getNeighborTypeSum(G2, 5))

    # Run Glauber Dynamics data switching simulation
    print("Running Glauber Dynamics Algorithm...")
    sampleTimes, energies, numGoodLinks, MixedGraph = GlauberDynamicsDataSwitch(G, times, 0.1, plot=False, samplingSize=10)
    # TODO: What temperature to define?
    # (t=0.5, n = 5000 for 40 nodes)
    # (t=0.1, n = 50000 for 100 nodes)
    # (t=0.05, n=750_000,sample=1000 for 1000 nodes)

    plotEnergy(sampleTimes, energies)
    plotGoodLinks(MixedGraph, sampleTimes, numGoodLinks)
    plotDiffHist(MixedGraph)
    MixedGraph.exportCSV("graphData/largeMixedGraph.csv")
    MixedGraph.plot_typed_graph("mixingPics/finalGraph.png")
    # Observation: Temperature is inversely proportional to rate of switching

    # TODO: Run Metropolis-Hastings data switching simulation
    #print("Running Metropolis-Hastings Algorithm...")
    #MetropolisHastingsDataSwitch(G, times, 1)

    print("Finished Simulations!")
    '''

    # --- Part 2: M-ary mixing ---
    print("Generating Graph...")
    G = mAryGraphGen(m=3, cluster_size=50, sparse_connections=5, p=0.5, path="graphData/mAryMixingGraph.csv", plotGraph=True)
    G.plot_typed_graph("mixingPics/startGraph.png", m=3)

    # Generate time points
    n = 100_000
    times = np.arange(n+1)

    # Run Glauber Dynamics M-ary data switching simulation
    sampleTimes, energies, numGoodLinks, MixedGraph = mAryGlauberDynamicsDataSwitch(3, G, times, 10, plot=False, samplingSize=100)
    plotEnergy(sampleTimes, energies)
    plotGoodLinks(MixedGraph, sampleTimes, numGoodLinks)
    plotDiffHist(MixedGraph)
    MixedGraph.plot_typed_graph("mixingPics/finalGraph.png", m=3)
