# Harris Ransom
# Iris Dataset Selection and Formatting
# 3/10/2025

# Imports
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler, Subset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define constants/parameters
trainsetPath = 'Iris_Data/trainset'
testsetPath = 'Iris_Data/testset'

# Load Iris dataset
# See: https://janakiev.com/blog/pytorch-iris/
iris = load_iris()
X = iris['data']
y = iris['target']
names = iris['target_names']
feature_names = iris['feature_names']

# Scale data to have mean 0 and variance 1 
# which is importance for convergence of the neural network
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data set into training and testing sets
# TODO: Verify split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=2)

# Visualize Iris data
def visualizeDataset():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    for target, target_name in enumerate(names):
        X_plot = X[y == target]
        ax1.plot(X_plot[:, 0], X_plot[:, 1], 
                linestyle='none', 
                marker='o', 
                label=target_name)
    ax1.set_xlabel(feature_names[0])
    ax1.set_ylabel(feature_names[1])
    ax1.axis('equal')
    ax1.legend()

    for target, target_name in enumerate(names):
        X_plot = X[y == target]
        ax2.plot(X_plot[:, 2], X_plot[:, 3], 
                linestyle='none', 
                marker='o', 
                label=target_name)
    ax2.set_xlabel(feature_names[2])
    ax2.set_ylabel(feature_names[3])
    ax2.axis('equal')
    ax2.legend()
    plt.show()
visualizeDataset()

# TODO: Format data into PyTorch dataset/dataloader
X_train = torch.Tensor(torch.from_numpy(X_train)).float()
y_train = torch.Tensor(torch.from_numpy(y_train)).long()
X_test  = torch.Tensor(torch.from_numpy(X_test)).float()
y_test  = torch.Tensor(torch.from_numpy(y_test)).long()

print(f"X_train dimensions: {X_train.size()}")
print(f"Y_train dimensions: {y_train.size()}")
print(f"X_test dimensions: {X_test.size()}")
print(f"Y_test diemsnions: {y_test.size()}")
print(y_train)
