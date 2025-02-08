# Harris Ransom
# Random Walk Learning Debugging w/Linear Regression
# 2/6/2025

# Importing Libraries
import torch
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for consistency
SEED = 45
torch.manual_seed(SEED)
np.random.seed(SEED)

# Generate pseudo-random synthetic data parameters
SIZE = 100
START_X = -5
END_X = 5
RANGE_LEN = END_X - START_X
X = torch.arange(START_X, END_X, RANGE_LEN/SIZE).view(-1, 1)
m_1 = np.random.uniform(-10, 10)
m_2 = np.random.uniform(-10, 10)
a_1 = np.random.uniform(-10, 10)
a_2 = np.random.uniform(-10, 10)
b_1 = np.random.uniform(-10, 10)
b_2 = np.random.uniform(-10, 10)
print(f"No. of data points: {2*SIZE}")
print(f"Class 1: Y_1 = {m_1}(X - {a_1}) + {b_1}")
print(f"Class 2: Y_2 = {m_2}(X - {a_2}) + {b_2}")

# Generate noisy synthetic data
noise_mean_1 = np.random.uniform(-10, 10)
noise_mean_2 = np.random.uniform(-10, 10)
variance = np.random.uniform(5, 10)
print(f"\nClass 1 Noise Mean: {noise_mean_1}")
print(f"Class 2 Noise Mean: {noise_mean_2}")
print(f"Noise Variance: {variance}")
Y_1 = m_1*(X-a_1) + b_1 + np.random.normal(noise_mean_1, variance, X.size())
Y_2 = m_2*(X-a_2) + b_2 + np.random.normal(noise_mean_2, variance, X.size())

# Plot synthetic data
plt.scatter(X, Y_1, label='Class 1')
plt.scatter(X, Y_2, label='Class 2')
plt.title("Synthetic Data")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

# Combine data into tensor objects
X = torch.cat((X-a_1, X-a_2))
Y = torch.cat((Y_1, Y_2))
labels = torch.cat((torch.zeros(SIZE), torch.ones(SIZE))).reshape(-1, 1)
print(f"X: {X.size()}")
print(f"Y: {Y.size()}")
print(f"Labels: {labels.size()}")

# TODO: Define linear regression model
class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)
