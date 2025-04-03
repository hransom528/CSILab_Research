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
from graphGen import graphGen, completeGraphGen, erdosRenyiGraphGen, dRegularGraphGen
from RandomWalk import MetropolisHastingsRandomWalk, plotTVDistances, plotRandomWalk
from DataMixing import GlauberDynamicsDataSwitch, plotEnergy, plotDiffHist, plotGoodLinks

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
#plotGeneratedData(X1, X2, Y1, Y2)

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
#plotTrainTestSplit(X_train, Y_train, X_test, Y_test)

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
LEARNING_RATE = 0.01
EPOCHS = 15000
RUNS = 50

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

    # Print model weights/parameters before training
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)

    # Iterate over n epochs for training/testing
    for i in range(EPOCHS):
        sample_ind = np.random.choice(np.arange(len(X)))
        X_sample = X_train[sample_ind]
        Y_sample = Y_train[sample_ind]
        train_loss = training(X_sample, Y_sample, model, loss_fn, optimizer)
        test_loss, accuracy = testing(X_test, Y_test, model)
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
#plotTestLosses(EPOCHS, averaged_centralized_test_losses)

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

# ------------------------------
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
G1 = completeGraphGen(2*train_size, path="./graphData/LinearRegressionGraph.csv", plotGraph=True)
#G2 = erdosRenyiGraphGen(2*train_size, 0.5, path="./graphData/LinearRegressionGraph.csv", plotGraph=True)
#G3 = dRegularGraphGen(2*train_size, 5, path="./graphData/LinearRegressionGraph.csv", plotGraph=True)
G4 = graphGen(train_size, 10, p=1, path="./graphData/LinearRegressionGraph.csv", plotGraph=True)
#print(G.edges) ~ 8000 edges

# Perform multiple random walk/training/testing runs
ITERATIONS = 7000
RUNS = 50
LEARNING_RATE = 0.01

print("Training and testing graph machine learning model...")
def graphRandomWalkLearn(G, X1_train, X2_train, Y1_train, Y2_train, X_test, Y_test, runs=5, timeCount=15000, plotResults=True, learning_rate=0.01):
    test_losses_runs = []
    accuracies_runs = []
    startingNode = np.random.choice(G.nodes) # Use the same starting node across each run
    for n in range(runs):
        # Performs random walk
        times = np.arange(timeCount)
        nodesVisited, P, tvDistances = MetropolisHastingsRandomWalk(G, times, startNode=startingNode)
        #plotRandomWalk(times, nodesVisited)

        # Set up machine learning model
        loss_fn = torch.nn.CrossEntropyLoss()
        model = LinearClassification()
        checkpoint = torch.load("starting_model.pth", weights_only=True)
        model.load_state_dict(checkpoint)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        # Perform machine learning based on nodes visited
        # Train at each node visited
        test_losses = []
        accuracies = []
        counter1 = 0
        counter2 = 0
        for i in range(len(nodesVisited)):
            nodeNum = nodesVisited[i]
            nodeType = G.getNodeType(nodeNum)
            #if (nodeNum > train_size-1):
            if (nodeType == 1):
                X_trainSample = X2_train[counter2]
                Y_trainSample = Y2_train[counter2]
                counter2 += 1
                counter2 %= len(X2_train)
            else:
                X_trainSample = X1_train[counter1]
                Y_trainSample = Y1_train[counter1]
                counter1 += 1
                counter1 %= len(X1_train)
            
            X_trainSample = torch.tensor(X_trainSample).type(torch.float)
            Y_trainSample = torch.tensor(Y_trainSample).type(torch.float)
            train_loss = training(X_trainSample, Y_trainSample, model, loss_fn, optimizer)

            # Test model on random sample
            X_test = torch.tensor(X_test).type(torch.float)
            Y_test = torch.tensor(Y_test).type(torch.float)
            test_loss, accuracy = testing(X_test, Y_test, model)

            # Get results from current node testing
            test_losses.append(test_loss)
            accuracies.append(accuracy)
            if (i % 100 == 0):
                print(f"\nIteration {i}: Training Loss = {train_loss.item()}")
                print(f"Iteration {i}: Testing Loss = {test_loss}")
                print(f"Iteration {i}: Accuracy = {accuracy}")

        # Append results of run
        test_losses_runs.append(test_losses)
        accuracies_runs.append(accuracies)

    # Average results across multiple runs
    averaged_test_losses = averageRunData(runs, test_losses_runs)
    averaged_accuracies = averageRunData(runs, accuracies_runs)

    # Output results from graph random walk learning
    if (plotResults):
        plotTestLosses(timeCount, averaged_test_losses, xlabel="Iterations")
        plotAccuracies(timeCount, averaged_accuracies, xlabel="Iterations")
    return averaged_test_losses, averaged_accuracies
averaged_complete_test_losses, averaged_complete_accuracies = graphRandomWalkLearn(G1, X1_train, X2_train, Y1_train, Y2_train, X_test, Y_test, runs=RUNS, timeCount=ITERATIONS, plotResults=False, learning_rate=LEARNING_RATE)
averaged_clustered_test_losses, averaged_clustered_accuracies = graphRandomWalkLearn(G4, X1_train, X2_train, Y1_train, Y2_train, X_test, Y_test, runs=RUNS, timeCount=ITERATIONS, plotResults=False, learning_rate=LEARNING_RATE)

# Test Erdos-Renyi GRW behavior at different p values
def pValueExperiment(runs, iterations, X1_train, X2_train, Y1_train, Y2_train, X_test, Y_test, plotResults=False):
    pTestLosses = []
    pAccuracies = []
    for p in np.arange(0.1, 1.1, 0.1):
        print(f"\nPerforming Graph Random Walk for Erdos-Renyi graph with p={p}...")
        G = erdosRenyiGraphGen(2*train_size, p, path="./graphData/LinearRegressionGraph.csv", plotGraph=False)
        averaged_test_losses, averaged_accuracies = graphRandomWalkLearn(G, X1_train, X2_train, Y1_train, Y2_train, X_test, Y_test, runs=runs, timeCount=iterations, plotResults=False)
        pTestLosses.append(averaged_test_losses)
        pAccuracies.append(averaged_accuracies)
    
    # Plot results (if configured)
    if (plotResults):
        plotTestLossesP(iterations, pTestLosses)
        plotAccuraciesP(iterations, pAccuracies)
    return pTestLosses, pAccuracies
# Plot averaged test losses and accuracies based on Erdos-Renyi p-value experiments
def plotTestLossesP(iterations, test_losses, xlabel="Iterations"):
    x = np.arange(iterations)
    plt.title("Averaged Test Losses over p")
    plt.xlabel(xlabel)
    plt.ylabel("Test Losses")
    plt.ylim([0, 1.05])
    for i in range(len(accuracies)):
        plt.plot(x, accuracies[i], label=f"p={(i+1)*0.1}")
    plt.legend()
    plt.show()
def plotAccuraciesP(iterations, accuracies, xlabel="Iterations"):
    x = np.arange(iterations)
    plt.title("Averaged Accuracies over p")
    plt.xlabel(xlabel)
    plt.ylabel("Accuracy")
    plt.ylim([0, 1.05])
    for i in range(len(accuracies)):
        plt.plot(x, accuracies[i], label=f"p={(i+1)*0.1}")
    plt.legend()
    plt.show()
#pTestLosses, pAccuracies = pValueExperiment(RUNS, ITERATIONS, X1_train, X2_train, Y1_train, Y2_train, X_test, Y_test, plotResults=False)
#plotTestLossesP(ITERATIONS, pTestLosses)
#plotAccuraciesP(ITERATIONS, pAccuracies)

# Test D-regular GRW behavior at different d values
def dRegularExperiment(runs, iterations, X1_train, X2_train, Y1_train, Y2_train, X_test, Y_test, plotResults=False):
    dTestLosses = []
    dAccuracies = []
    for d in np.arange(1, 11):
        print(f"\nPerforming Graph Random Walk for D-regular graph with d={d}...")
        G = dRegularGraphGen(2*train_size, d, path="./graphData/LinearRegressionGraph.csv", plotGraph=False)
        averaged_test_losses, averaged_accuracies = graphRandomWalkLearn(G, X1_train, X2_train, Y1_train, Y2_train, X_test, Y_test, runs=runs, timeCount=iterations, plotResults=False)
        dTestLosses.append(averaged_test_losses)
        dAccuracies.append(averaged_accuracies)
    
    # Plot results (if configured)
    if (plotResults):
        plotTestLossesP(iterations, dTestLosses)
        plotAccuraciesP(iterations, dAccuracies)
    return dTestLosses, dAccuracies
# Plot averaged test losses and accuracies based on D-regular experiments
def plotTestLossesD(iterations, test_losses, xlabel="Iterations"):
    x = np.arange(iterations)
    plt.title("Averaged Test Losses over d")
    plt.xlabel(xlabel)
    plt.ylabel("Test Losses")
    plt.ylim([0, 1.05])
    for i in range(len(test_losses)):
        plt.plot(x, test_losses[i], label=f"d={(i+1)}")
    plt.legend()
    plt.show()
def plotAccuraciesD(iterations, accuracies, xlabel="Iterations"):
    x = np.arange(iterations)
    plt.title("Averaged Accuracies over d")
    plt.xlabel(xlabel)
    plt.ylabel("Accuracy")
    plt.ylim([0, 1.05])
    for i in range(len(accuracies)):
        plt.plot(x, accuracies[i], label=f"d={(i+1)}")
    plt.legend()
    plt.show()
#dTestLosses, dAccuracies = dRegularExperiment(RUNS, ITERATIONS, X1_train, X2_train, Y1_train, Y2_train, X_test, Y_test, plotResults=False)
#plotTestLossesD(ITERATIONS, dTestLosses)
#plotAccuraciesD(ITERATIONS, dAccuracies)

# Mix graph
n = 75_000
times = np.arange(n+1)
sampleTimes, energies, numGoodLinks, Gmixed = GlauberDynamicsDataSwitch(G4, times, 0.1, plot=False, samplingSize=100)
plotDiffHist(Gmixed)

# Perform random walk on mixed graph
averaged_mixed_test_losses, averaged_mixed_accuracies = graphRandomWalkLearn(Gmixed, X1_train, X2_train, Y1_train, Y2_train, X_test, Y_test, runs=RUNS, timeCount=ITERATIONS, plotResults=False, learning_rate=LEARNING_RATE)

# Plot combined graphs from centralized, complete, and clustered runs
def plotCombinedTestLosses(iterations, centralized_test_losses, complete_test_losses, clustered_test_losses, mixed_test_losses, xlabel="Iterations"):
    x = np.arange(iterations)
    plt.title("Comparison of Learning Test Losses")
    plt.xlabel(xlabel)
    plt.ylabel("Test Losses")
    plt.ylim([0, 1.05])

    plt.plot(x, centralized_test_losses, label="Centralized")
    plt.plot(x, complete_test_losses, label="Complete Graph")
    plt.plot(x, clustered_test_losses, label="Clustered Erdos-Renyi (Unmixed)")
    plt.plot(x, mixed_test_losses, label="Clustered Erdos-Renyi (Mixed)")
    plt.legend()
    plt.show()
def plotCombinedAccuracies(iterations, centralized_accuracies, complete_accuracies, clustered_accuracies, mixed_accuracies, xlabel="Iterations"):
    x = np.arange(iterations)
    plt.title("Comparison of Learning Test Accuracies")
    plt.xlabel(xlabel)
    plt.ylabel("Test Accuracy")
    plt.ylim([0, 1.05])

    plt.plot(x, centralized_accuracies, label="Centralized")
    plt.plot(x, complete_accuracies, label="Complete Graph")
    plt.plot(x, clustered_accuracies, label="Clustered Erdos-Renyi (Unmixed)")
    plt.plot(x, mixed_accuracies, label="Clustered Erdos-Renyi (Mixed)")
    plt.legend()
    plt.show()
plotCombinedAccuracies(ITERATIONS, averaged_centralized_accuracies, averaged_complete_accuracies, averaged_clustered_accuracies, averaged_mixed_accuracies)