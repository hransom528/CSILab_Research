# Harris Ransom
# Total Variational Distance Calculator
# 9/10/2024

# Imports
import numpy as np

# Function to calculate the total variational distance between two probability distributions
def tvDistance(pi, mu):
	# Calculate the total variational distance between two probability distributions
	# pi and mu
	# Inputs:
	# pi - the first probability distribution
	# mu - the second probability distribution
	# Outputs:
	# tvDist - the total variational distance between pi and mu

	# Initialize the total variational distance
	tvDist = 0

	# Check if the two probability distributions are the same length
	if (len(pi) != len(mu)):
		raise ValueError("The two probability distributions must be the same length")

	# Iterate through the probability distributions
	for i in range(len(pi)):
		# Calculate the absolute difference between the two probabilities
		diff = np.abs(pi[i] - mu[i])

		# Update the total variational distance
		tvDist += diff

	# Return the total variational distance
	tvDist = 0.5 * tvDist
	return tvDist

# MAIN
if __name__ == "__main__":
	# TODO: Get input
	stationaryDist = []
	measuredDist = []

	# Check distributions
	if (sum(stationaryDist) != 1):
		raise ValueError("The stationary distribution must sum to 1")
	if (sum(measuredDist) != 1):
		raise ValueError("The measured distribution must sum to 1")
	
	# Calculate the total variational distance
	tvDist = tvDistance(stationaryDist, measuredDist)
	print("The total variational distance is: ", tvDist)
