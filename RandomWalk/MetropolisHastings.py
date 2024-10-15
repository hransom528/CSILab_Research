# Harris Ransom
# Metropolis-Hastings Algorithm Implementation
# See: https://prappleizer.github.io/Tutorials/MetropolisHastings/MetropolisHastings_Tutorial.html

# Imports
import numpy as np
from Graph import Graph
from scipy import stats

# Metropolis Hastings class
class MetropolisHastings():
    # Constructor
    def __init__(self):
        self.Uniform = stats.uniform()

    def initialize(self, theta_init, nsteps):
        self.theta_init = theta_init
        self.nsteps = nsteps

    def set_proposal_sigma(self, sigma):
        covMatrix = sigma^2 * np.eye(self.nodes) # TODO: Fix dimension
        self.proposal_distribution = stats.multivariate_normal(mean=None, cov=covMatrix)

    def set_target_distribution(self, pi):
        pass

    def sample():
        pass

# MAIN
if __name__ == "__main__":
    # Create a graph object
    G = Graph.importCSV("graphData/simulation.csv")
    print(G)
        
    # Create a stationary distribution
    #pi = np.ones(G.nodes) / G.nodes # Uniform
    pi = [G.getDegree(i+1) for i in range(0, G.nodes)]
    print(pi)

    # Run the Metropolis-Hastings algorithm
    #MetropolisHastings(G, pi)