# Harris Ransom
# Random Walk Simulation

# Imports
import numpy as np
import matplotlib.pyplot as plt
from Graph import Graph
from MetropolisHastings import MetropolisHastings

# Plots random walk simulation
def plotRandomWalk(times, nodesVisited):
    plt.plot(times, nodesVisited)
    plt.title("Random Walk Simulation")
    plt.xlabel("Time Steps")
    plt.ylabel("Node Visited")
    plt.show()

# MAIN
if __name__ == "__main__":
    pass
    # Create a graph object
    G = Graph.importCSV("simulation.csv")
    print(G)

    # Generate time points
    n = 500
    times = np.arange(n+1)
    
    # Create a stationary distribution
    sd = G.getStationaryDistribution()
    #print(sd)

    # Metropolis Hastings
    S = MetropolisHastings(G, sd)

    # Perform the random walk
    nodesVisited = []
    initialNode = np.random.choice(G.nodes) # Randomly choose initial node
    for t in times:
        # Randomly choose a node to travel to from current node
        if t == 0:
            currentNode = initialNode
        else:
            currentNode = np.random.choice(np.arange(G.nodes), p=S[currentNode, :])
        
        nodesVisited.append(currentNode)
    
    # Output simulation results
    plotRandomWalk(times, nodesVisited)
