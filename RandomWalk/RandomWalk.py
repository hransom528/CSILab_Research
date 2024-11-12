# Harris Ransom
# Random Walk Simulation

# Imports
import numpy as np
import matplotlib.pyplot as plt
from Graph import Graph
from TVDistance import tvDistance

# Plots random walk simulation
def plotRandomWalk(times, nodesVisited):
    plt.plot(times, nodesVisited)
    plt.title("Random Walk Simulation")
    plt.xlabel("Time Steps")
    plt.ylabel("Node Visited")
    plt.show()

# Plots random walk TV Distance vs. Time
def plotTVDistances(times, tvDistances):
    plt.plot(times, tvDistances)
    plt.title("Random Walk Simulation")
    plt.xlabel("Time Steps")
    plt.ylabel("TV Distance")
    plt.show()

# Lazy Random Walk function
# G - Graph
# pi - Stationary distribution
# TODO: Finish this function
def SimpleLazyRandomWalk(G, pi, times):
    # 1.) Get an initial proposed transition matrix
    n = G.nodes
    Q = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            if (G.A[i, j] > 0) or (i == j):
                Q[i, j] = 1 / (G.getDegree(i) + 1)
            else:
                Q[i, j] = 0

    # 2.) Check Q for probability distribution criteria
    for i in range(0, n):
        if (np.sum(Q[i, :]) != 1):
            raise ValueError(f"Q[{i}, :] is not a probability distribution")
    print(Q)

    # TODO: Finish the rest of the algorithm

    # TODO: Return output
    return Q

# Random Walk function
# G - Graph
# pi - Stationary distribution
def MetropolisHastingsRandomWalk(G, times):
    # 1.) Get an initial proposed transition matrix
    n = G.nodes
    Q = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            if (G.A[i, j] != 0):
                Q[i, j] = 1 / G.getDegree(i)
            else:
                Q[i, j] = 0

    # 2.) Check Q for probability distribution criteria
    for i in range(0, n):
        sumQ = np.sum(Q[i, :])
        if (abs(sumQ - 1) > 0.05):
            raise ValueError(f"Q[{i}, :] is not a probability distribution (sum = {sumQ})")
    #print(Q)

    # 3.) Calculates P from Q and pi
    P = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            if (i != j):
                if (Q[i, j] != 0) and (Q[j, i] != 0):
                    P[i, j] = Q[i, j] * min(1, (Q[j, i] / Q[i, j]))
                else:
                    P[i, j] = 0
    for i in range(0, n):
        P[i, i] = 1 - np.sum(P[i, :])

    # 4.) Perform Metropolis-Hastings Random Walk
    nodesVisited = []
    tvDistances = []
    initialNode = np.random.choice(G.nodes) # Randomly choose initial node
    for t in times:
        # Randomly choose a node to travel to from current node
        if t == 0:
            currentNode = initialNode
        else:
            #neighborSet = G.getNeighborSet(currentNode)
            currentNode = np.random.choice(np.arange(G.nodes), p=P[currentNode, :])
        
        # Add node to nodes visited
        nodesVisited.append(currentNode)
        
        # Get node counts and frequencies
        unique, nodeCounts = np.unique(nodesVisited, return_counts=True)
        frequencies = np.zeros(G.nodes)
        for i in range(0, len(unique)):
            frequencies[unique[i]] = nodeCounts[i] / (t+1)

        # Calculate TV distance between stationary distribution and frequencies
        #sd = G.getStationaryDistribution()
        sd = [1/G.nodes] * G.nodes
        tvDist = tvDistance(sd, frequencies)
        tvDistances.append(tvDist)

    # Return output
    return nodesVisited, P, tvDistances

# MAIN
if __name__ == "__main__":
    pass
    # Create a graph object
    G = Graph.importCSV("graphData/simulation.csv")
    print(G)

    # Generate time points
    n = 10000
    times = np.arange(n+1)
    
    # Create a (simple random walk) stationary distribution
    #sd = G.getStationaryDistribution()
    #print(sd)

    # Perform Metropolis Hastings Random Walk
    nodesVisited, P, tvDistances = MetropolisHastingsRandomWalk(G, times)
    #print(nodesVisited)
    
    # Output simulation results
    #plotRandomWalk(times, nodesVisited)
    plotTVDistances(times, tvDistances)
