# Harris Ransom
# Random Walk Learning Debugging w/Linear Classification
# 2/6/2025

# Importing Libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Custom imports
from Graph import Graph
from graphGen import graphGen, completeGraphGen
from RandomWalk import MetropolisHastingsRandomWalk, plotTVDistances, plotRandomWalk

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

# Combine data into tensor/dataset objects
dataset1 = torch.tensor([X1, Y1, np.ones(SIZE)]).T
dataset2 = torch.tensor([X2, Y2, 2*np.ones(SIZE)]).T
dataset = torch.cat((dataset1, dataset2))
#print(dataset.shape)
#print(dataset)

# Helper function to shuffle data
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

# Transform dataset into feature/label tensors
coords = np.zeros((2*SIZE, 2))
for i in range(0, 2*SIZE):
    coords[i] = [dataset[i, 0], dataset[i, 1]]
coords = torch.tensor(coords)
prior_labels = dataset[:, 2]
coords, prior_labels = unison_shuffled_copies(coords, prior_labels)

# Encode labels as one-hot
labels = np.zeros((2*SIZE, 2))
for i in range(len(prior_labels)):
    if (prior_labels[i] == 1):
        labels[i] = [1, 0]
    elif (prior_labels[i] == 2):
        labels[i] = [0, 1]
labels = torch.tensor(labels)

# Convert tensor datatypes
coords = coords.to(torch.float)
labels = labels.to(torch.float)

# Train-test split
SIZE = 2*SIZE # Accounts for both classes
PERCENT_TRAIN = 0.75
train_size = int(PERCENT_TRAIN * SIZE)
test_size = SIZE - train_size

X_train = coords[:train_size]
Y_train = labels[:train_size]
X_test = coords[train_size:]
Y_test = labels[train_size:]
print(f"\nTrain Size: {train_size}")
print(f"Test Size: {test_size}")
print(f"X_train: {X_train.size()}")
print(f"Y_train: {Y_train.size()}")
print(f"X_test: {X_test.size()}")
print(f"Y_test: {Y_test.size()}")

# Plot train-test split
def plotTrainTestSplit(X_train, Y_train, X_test, Y_test):
    plt.figure()
    plt.scatter(X_train[:, 0], X_train[:, 1], label='Train Data')
    plt.scatter(X_test[:, 0], X_test[:, 1], label='Test Data')
    plt.title("Train-Test Split")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()
plotTrainTestSplit(X_train, Y_train, X_test, Y_test)

# Define linear classification model
# See: https://www.geeksforgeeks.org/linear-regression-using-pytorch/
# https://www.geeksforgeeks.org/classification-using-pytorch-linear-function/
class LinearClassification(torch.nn.Module):
    def __init__(self):
        super(LinearClassification, self).__init__()
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(2, 2),
            torch.nn.Softmax()
        )
        
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
LEARNING_RATE = 0.1
EPOCHS = 2500
RUNS = 5#10
#loss_fn = torch.nn.MSELoss(reduction='sum')
loss_fn = torch.nn.CrossEntropyLoss()
model = LinearClassification()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

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
def testing(X, Y_test, model):
    if (Y_test.size() == 0):
        raise ValueError("No test data provided (length 0).")
    if (Y_test.size() != X.size()):
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

# Perform training and testing for multiple epochs
print("\nTraining and testing model...")
test_losses_runs = []
accuracies_runs = []
for i in range(RUNS):
    # Reset model/data in between runs
    test_losses = []
    accuracies = []
    loss_fn = torch.nn.CrossEntropyLoss()
    model = LinearClassification()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # Iterate over n epochs for training/testing
    for i in range(EPOCHS):
        train_loss = training(X_train, Y_train, model, loss_fn, optimizer)
        test_loss, accuracy = testing(X_test, Y_test, model)
        test_losses.append(test_loss)
        accuracies.append(accuracy)
        if (i % 100 == 0):
            print(f"\nEpoch {i}: Training Loss = {train_loss.item()}")
            print(f"Epoch {i}: Testing Loss = {test_loss.item()}")
            print(f"Epoch {i}: Accuracy = {accuracy}")
    
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
averaged_test_losses = averageRunData(RUNS, test_losses_runs)
averaged_accuracies = averageRunData(RUNS, accuracies_runs)

# Plots test losses over time (epochs)
def plotTestLosses(epochs, test_losses):
    x = np.arange(epochs)
    plt.plot(x, test_losses)
    plt.title("Testing Losses over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Testing Loss (CrossEntropy)")
    plt.ylim([0, 1])
    plt.show()
plotTestLosses(EPOCHS, averaged_test_losses)

# Plots accuracies over time (epochs)
def plotAccuracies(epochs, accuracies):
    x = np.arange(epochs)
    plt.plot(x, accuracies)
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.ylim([0, 1.05])
    plt.show()
plotAccuracies(EPOCHS, averaged_accuracies)

# Part 2: Graph machine learning
# Get training dataset
X1, Y1 = unison_shuffled_copies(X1, Y1)
X2, Y2 = unison_shuffled_copies(X2, Y2)
train_size = int(train_size // 2) # Divide into two groups for splitting
X1_train = np.array(X1[:train_size]).reshape((train_size, 1))
X2_train = np.array(X2[:train_size]).reshape((train_size, 1))
Y1_train = np.array(Y1[:train_size]).reshape((train_size, 1))
Y2_train = np.array(Y2[:train_size]).reshape((train_size, 1))

# Combine X/Y coords, add labels for training dataset
X1_train = np.concatenate((X1_train, Y1_train), axis=1)
X2_train = np.concatenate((X2_train, Y2_train), axis=1)
Y1_train = np.zeros((train_size, 2))
Y2_train = np.zeros((train_size, 2))
for i in range(train_size):
    Y1_train[i] = [1, 0]
    Y2_train[i] = [0, 1]

# Get testing dataset
X1_test = X1[train_size:]
X2_test = X2[train_size:]
Y1_test = Y1[train_size:]
Y2_test = Y2[train_size:]
X_test = [*X1_test, *X2_test]
Y_test = [*Y1_test, *Y2_test]
test_size = len(X_test)

# Convert testing dataset to (coords, labels) format
# Labels are one-hot encoded
X_test = np.array(X_test).reshape((test_size, 1))
Y_test = np.array(Y_test).reshape((test_size,1))
X_test = np.concatenate((X_test, Y_test), axis=1) # Combine X,Y into coords
Y_test = np.zeros((test_size, 2))
for i in range(len(X1_test)):
    Y_test[i] = [1, 0]
for i in range(len(X1_test), len(X_test)):
    Y_test[i] = [0, 1]
X_test, Y_test = unison_shuffled_copies(X_test, Y_test)

# Generates graph structure
G = completeGraphGen(2*train_size, path="./graphData/LinearRegressionGraph.csv", plotGraph=True)
#graphGen(train_size, 10, p=1, path="./graphData/LinearRegressionGraph.csv", plotGraph=True)
#print(G.edges) ~ 8000 edges

# Perform multiple random walk/training/testing runs
print("Training and testing graph machine learning model...")
RUNS = 5
TIMES = 15000
test_losses_runs = []
accuracies_runs = []
startingNode = np.random.choice(G.nodes) # Use the same starting node across each run
for n in range(RUNS):
    # Performs random walk
    times = np.arange(TIMES)
    nodesVisited, P, tvDistances = MetropolisHastingsRandomWalk(G, times, startNode=startingNode)
    #plotRandomWalk(times, nodesVisited)

    # Set up machine learning model
    loss_fn = torch.nn.CrossEntropyLoss()
    model = LinearClassification()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # Perform machine learning based on nodes visited
    # Train at each node visited
    test_losses = []
    accuracies = []
    for i in range(len(nodesVisited)):
        nodeNum = nodesVisited[i]
        if (nodeNum > train_size-1):
            nodeNum -= train_size
            X_trainSample = X2_train[nodeNum]
            Y_trainSample = Y2_train[nodeNum]
        else:
            X_trainSample = X1_train[nodeNum]
            Y_trainSample = Y1_train[nodeNum]
        
        X_trainSample = torch.tensor(X_trainSample).type(torch.float)
        Y_trainSample = torch.tensor(Y_trainSample).type(torch.float)
        train_loss = training(X_trainSample, Y_trainSample, model, loss_fn, optimizer)

        # Test model on random sample
        test_loss_average = 0
        accuracy_average = 0
        X_test = torch.tensor(X_test).type(torch.float)
        Y_test = torch.tensor(Y_test).type(torch.float)
        test_loss, accuracy = testing(X_test, Y_test, model)

        # Get results from current node testing
        test_losses.append(test_loss)
        accuracies.append(accuracy)
        if (i % 100 == 0):
            print(f"\nEpoch {i}: Training Loss = {train_loss.item()}")
            print(f"Epoch {i}: Testing Loss = {test_loss}")
            print(f"Epoch {i}: Accuracy = {accuracy}")

    # Append results of run
    test_losses_runs.append(test_losses)
    accuracies_runs.append(accuracies)

# Average results across multiple runs
averaged_test_losses = averageRunData(RUNS, test_losses_runs)
averaged_accuracies = averageRunData(RUNS, accuracies_runs)

# Output results from graph random walk learning
plotTestLosses(TIMES, averaged_test_losses)
plotAccuracies(TIMES, averaged_accuracies)