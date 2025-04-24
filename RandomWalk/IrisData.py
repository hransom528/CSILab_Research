# Harris Ransom
# Iris Dataset Selection and Formatting
# 3/10/2025

# Imports
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler, Subset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Custom Imports
from Graph import Graph
from graphGen import graphGen
from TVDistance import tvDistance
from RandomWalk import MetropolisHastingsRandomWalk, plotTVDistances
from DataMixing import GlauberDynamicsDataSwitch, plotEnergy, plotDiffHist, plotGoodLinks
from MNISTData import loadMNISTData

# Set random seed for consistency
SEED = 45
torch.manual_seed(SEED)
np.random.seed(SEED)

# Define constants/parameters
trainsetPath = 'Iris_Data/trainset'
testsetPath = 'Iris_Data/testset'

# Load Iris dataset
# See: https://janakiev.com/blog/pytorch-iris/
iris = load_iris()
X = iris['data']
y = iris['target']
SIZE = len(y)
print(SIZE)
names = iris['target_names']
feature_names = iris['feature_names']

# Scale data to have mean 0 and variance 1 
# which is important for convergence of the neural network
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
#visualizeDataset()

# Format data into PyTorch dataset/dataloader
X_train = torch.Tensor(torch.from_numpy(X_train)).float()
y_train = torch.Tensor(torch.from_numpy(y_train)).long()
X_test  = torch.Tensor(torch.from_numpy(X_test)).float()
y_test  = torch.Tensor(torch.from_numpy(y_test)).long()

print(f"X_train dimensions: {X_train.size()}")
print(f"Y_train dimensions: {y_train.size()}")
print(f"X_test dimensions: {X_test.size()}")
print(f"Y_test diemsnions: {y_test.size()}")

# Encode labels as one-hot
def oneHotLabel(label_in):
    if (label_in == 0):
        return [0, 0, 1]
    elif (label_in == 1):
        return [0, 1, 0]
    elif (label_in == 2):
        return [1, 0, 0]
    else:
        return [0, 0, 0]
trainLabels = []
testLabels = []
for i in range(len(y_train)):
    trainLabels.append(oneHotLabel(y_train[i]))
for i in range(len(y_test)):
    testLabels.append(oneHotLabel(y_test[i]))
y_train = torch.Tensor(trainLabels)
y_test = torch.Tensor(testLabels)

# Define linear classification model
# See: https://www.geeksforgeeks.org/linear-regression-using-pytorch/
# https://www.geeksforgeeks.org/classification-using-pytorch-linear-function/
class LinearClassification(torch.nn.Module):
    def __init__(self):
        super(LinearClassification, self).__init__()
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(4, 3),
            torch.nn.Softmax()
        )
        
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
LEARNING_RATE = 0.01
EPOCHS = 10000
RUNS = 5

# Train the model
def training(X, Y_train, model, loss_fn, optimizer):
    # Forward pass: Compute predicted y by passing x to the model
    Y_pred = model(X)

    # Compute and print loss
    loss = loss_fn(Y_pred, Y_train)

    # Zero gradients, perform a backward pass, and update the weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss
    
# Test the model
def testing(X, Y_test, loss_fn, model):
    if (Y_test.size() == 0):
        raise ValueError("No test data provided (length 0).")
    if (len(Y_test) != len(X)):
        print(len(X))
        print(len(Y_test))
        raise ValueError("X and Y_test must be the same size.")
    
    with torch.no_grad():
        Y_pred = model(X)
        loss = loss_fn(Y_pred, Y_test)

        count = 0
        for i in range(len(Y_pred)):
            if (torch.equal(torch.round(Y_pred[i]), Y_test[i])):
                count += 1
        accuracy = count / len(Y_test)
    return loss, accuracy

# Perform training and testing for multiple runs
print("\nTraining and testing model...")
test_losses_runs = []
accuracies_runs = []

starting_model = LinearClassification()
starting_parameters = starting_model.parameters()
torch.save(starting_model.state_dict(), "starting_model.pth")
for i in range(RUNS):
    # Reset model/data in between runs
    test_losses = []
    accuracies = []

    model = LinearClassification()
    loss_fn = torch.nn.CrossEntropyLoss()
    checkpoint = torch.load("starting_model.pth", weights_only=True)
    model.load_state_dict(checkpoint)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # Iterate over n epochs for training/testing
    for i in range(EPOCHS):
        sample_ind = np.random.choice(np.arange(len(X_train)))
        X_sample = X_train[sample_ind]
        Y_sample = y_train[sample_ind]
        train_loss = training(X_sample, Y_sample, model, loss_fn, optimizer)
        test_loss, accuracy = testing(X_test, y_test, loss_fn, model)
        test_losses.append(test_loss)
        accuracies.append(accuracy)
        if (i % 1000 == 0):
            print(f"\nIteration {i}: Training Loss = {train_loss.item()}")
            print(f"Iteration {i}: Testing Loss = {test_loss.item()}")
            print(f"Iteration {i}: Accuracy = {accuracy}")
        
    # Save run results to 2D arrays
    test_losses_runs.append(test_losses)
    accuracies_runs.append(accuracies)

# Perform averaging of run data
def averageRunData(runs, data):
    epochs = len(data[0])
    averaged_data = []
    for i in range(epochs):
        average = 0
        for j in range(runs):
            average += data[j][i]
        average /= runs
        averaged_data.append(average)
    return averaged_data
averaged_centralized_test_losses = averageRunData(RUNS, test_losses_runs)
averaged_centralized_accuracies = averageRunData(RUNS, accuracies_runs)

# Plots test losses over time (epochs)
def plotTestLosses(epochs, test_losses, xlabel="Epochs"):
    x = np.arange(epochs)
    plt.plot(x, test_losses)
    plt.title("Averaged Testing Losses")
    plt.xlabel(xlabel)
    plt.ylabel("Testing Loss (CrossEntropy)")
    plt.ylim([0, 1])
    plt.show()
plotTestLosses(EPOCHS, averaged_centralized_test_losses)

# Plots accuracies over time (epochs)
def plotAccuracies(epochs, accuracies, xlabel="Epochs"):
    x = np.arange(epochs)
    plt.plot(x, accuracies)
    plt.title("Averaged Accuracies")
    plt.xlabel(xlabel)
    plt.ylabel("Accuracy")
    plt.ylim([0, 1.05])
    plt.show()
plotAccuracies(EPOCHS, averaged_centralized_accuracies)
