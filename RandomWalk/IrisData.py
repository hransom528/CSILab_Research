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

# TODO: Visualize data

# TODO: Format data into PyTorch dataset/dataloader