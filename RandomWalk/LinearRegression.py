# Harris Ransom
# Random Walk Learning Debugging w/Linear Regression
# 2/15/2025

# Importing Libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Set random seed for consistency
SEED = 45
torch.manual_seed(SEED)
np.random.seed(SEED)

# Generate pseudo-random synthetic data parameters
SIZE = 100
mu1 = np.random.uniform(-10, 10)
mu2 = np.random.uniform(-10, 10)
mu = np.array([mu1, mu2])
covariance = np.zeros((2,2))
with np.nditer(covariance, op_flags=['readwrite']) as it:
    for i in it:
        i[...] = np.random.uniform(3, 10)

X = np.random.multivariate_normal(mu, covariance, SIZE)
X1 = np.zeros(SIZE)
X2 = np.zeros(SIZE)
for i in range(0, len(X)):
    X1[i] = X[i][0]
    X2[i] = X[i][1]

beta1 = np.random.uniform(-10, 10)
beta2 = np.random.uniform(-10, 10)
e1 = np.random.normal(0, 1, SIZE)
e2 = np.random.normal(0, 1, SIZE)

Y1 = beta1*X1 + e1
Y2 = beta2*X2 + e2

# Output relevant parameters
print(f"No. of data points per cluster: {SIZE}")
print(f"Class 1: Y_1 = {beta1}(X) + e1")
print(f"Class 2: Y_2 = {beta2}(X) + e2")
print(f"Covariance Matrix: {covariance}")

# Plot synthetic data
def plotGeneratedData(X1, X2, Y1, Y2):
    plt.figure()
    plt.scatter(X1, Y1, label='Class 1')
    plt.scatter(X2, Y2, label='Class 2')
    plt.title("Synthetic Data")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()
plotGeneratedData(X1, X2, Y1, Y2)