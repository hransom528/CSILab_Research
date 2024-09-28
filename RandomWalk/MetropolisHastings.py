# Harris Ransom
# Metropolis-Hastings Algorithm Implementation

# Imports
import numpy as np
from Graph import Graph

# Metropolis-Hastings function
# G - Graph
# pi - Stationary distribution
def MetropolisHastings(G, pi):
    # 1.) Get an initial proposed transition matrix
    n = G.nodes
    Q = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            if (G.A[i, j] > 0):
                Q[i, j] = 1 / G.getDegree(i+1)
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

# MAIN
if __name__ == "__main__":
    # Create a graph object
    G = Graph.importCSV("simulation.csv")
    print(G)
        
    # Create a stationary distribution
    #pi = np.ones(G.nodes) / G.nodes # Uniform
    pi = [G.getDegree(i+1) for i in range(0, G.nodes)]
    print(pi)

    # Run the Metropolis-Hastings algorithm
    #MetropolisHastings(G, pi)