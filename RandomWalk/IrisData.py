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
from graphGen import graphGen, mAryGraphGen, mAryCompleteGraphGen
from TVDistance import tvDistance
from RandomWalk import MetropolisHastingsRandomWalk, plotTVDistances
from DataMixing import GlauberDynamicsDataSwitch, mAryGlauberDynamicsDataSwitch, plotEnergy, plotDiffHist, plotGoodLinks, plotTVHist
from MNISTData import loadMNISTData

# Set plot font size
import matplotlib
matplotlib.rcParams.update({'font.size': 22})

# Save results to/from files
def saveArrToFile(arr, path="results/arr0.txt"):
    with open(path, "w") as output:
        outStr = str(arr)
        outStr = outStr.replace("[", "")
        outStr = outStr.replace("]", "")
        output.write(outStr)
def loadArrFromFile(path="results/arr0.txt"):
    arr = []
    with open(path, "r") as file:
        arrStr = file.readline()
        arr = arrStr.split(",")
        arr = [float(i) for i in arr]
    return arr

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
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X)

# Split the data set into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2)

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
EPOCHS = 10_000
RUNS = 50
LEARNING_RATE = 0.05

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
torch.save(starting_model.state_dict(), "starting_model_iris.pth")
for i in range(RUNS):
    # Reset model/data in between runs
    test_losses = []
    accuracies = []

    model = LinearClassification()
    loss_fn = torch.nn.CrossEntropyLoss()
    checkpoint = torch.load("starting_model_iris.pth", weights_only=True)
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
saveArrToFile(averaged_centralized_accuracies, path="results/iris2/centralized.txt")

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
#plotAccuracies(EPOCHS, averaged_centralized_accuracies)

# ------------------------------
# Part 2: Graph machine learning

# Format dataset for graph training
CLUSTER_SIZE = 40
TEST_SIZE = 50 - CLUSTER_SIZE
X1 = []
X2 = []
X3 = []
Y1 = []
Y2 = []
Y3 = []
for i in range(len(X)):
    if (y[i] == 0):
        X1.append(X[i])
        Y1.append(y[i])
    elif (y[i] == 1):
        X2.append(X[i])
        Y2.append(y[i])
    elif (y[i] == 2):
        X3.append(X[i])
        Y3.append(y[i])
X1_train = X1[:CLUSTER_SIZE]
X2_train = X2[:CLUSTER_SIZE]
X3_train = X3[:CLUSTER_SIZE]
Y1_train = Y1[:CLUSTER_SIZE]
Y2_train = Y2[:CLUSTER_SIZE]
Y3_train = Y3[:CLUSTER_SIZE]

X1_test = X1[CLUSTER_SIZE:]
X2_test = X2[CLUSTER_SIZE:]
X3_test = X3[CLUSTER_SIZE:]
Y1_test = Y1[CLUSTER_SIZE:]
Y2_test = Y2[CLUSTER_SIZE:]
Y3_test = Y3[CLUSTER_SIZE:]
X_test = [*X1_test, *X2_test, *X3_test]
Y_test = [*Y1_test, *Y2_test, *Y3_test]

# One-hot encode labels
trainLabels = []
for i in range(len(Y1_train)):
    trainLabels.append(oneHotLabel(Y1_train[i]))
Y1_train = torch.Tensor(trainLabels)

trainLabels = []
for i in range(len(Y2_train)):
    trainLabels.append(oneHotLabel(Y2_train[i]))
Y2_train = torch.Tensor(trainLabels)

trainLabels = []
for i in range(len(Y3_train)):
    trainLabels.append(oneHotLabel(Y3_train[i]))
Y3_train = torch.Tensor(trainLabels)

testLabels = []
for i in range(len(Y_test)):
    testLabels.append(oneHotLabel(Y_test[i]))
Y_test = torch.Tensor(testLabels)

# Generate graph structure
G = mAryGraphGen(m=3, cluster_size=CLUSTER_SIZE, sparse_connections=5, p=0.3, path="graphData/generatedMAryClusteredGraph.csv", plotGraph=False)
G2 = mAryCompleteGraphGen(3*CLUSTER_SIZE, m=3, path="./graphData/generatedMAryCompleteGraph.csv", plotGraph=False)
# Perform multiple random walk/training/testing runs
ITERATIONS = 10_000
RUNS = 50
LEARNING_RATE = 0.05

print("Training and testing graph machine learning model...")
def graphRandomWalkLearn(G, X1_train, X2_train, X3_train, Y1_train, Y2_train, Y3_train, X_test, Y_test, 
                         runs=5, timeCount=15000, plotResults=False, learning_rate=0.01):
    test_losses_runs = []
    accuracies_runs = []
    startingNode = np.random.choice(G.nodes) # Use the same starting node across each run
    for n in range(runs):
         # Performs random walk
        times = np.arange(timeCount)
        nodesVisited, P, tvDistances = MetropolisHastingsRandomWalk(G, times, startNode=startingNode)

        # Set up machine learning model
        loss_fn = torch.nn.CrossEntropyLoss()
        model = LinearClassification()
        checkpoint = torch.load("starting_model_iris.pth", weights_only=True)
        model.load_state_dict(checkpoint)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        # Perform machine learning based on nodes visited
        # Train at each node visited
        test_losses = []
        accuracies = []
        counter1 = 0
        counter2 = 0
        counter3 = 0
        for i in range(len(nodesVisited)):
            nodeNum = nodesVisited[i]
            nodeType = G.getNodeType(nodeNum)
            if (nodeType == 1):
                X_trainSample = X1_train[counter1]
                Y_trainSample = Y1_train[counter1]
                counter1 += 1
                counter1 %= len(X1_train)
            elif (nodeType == 2):
                X_trainSample = X2_train[counter2]
                Y_trainSample = Y2_train[counter2]
                counter2 += 1
                counter2 %= len(X2_train)
            else:
                X_trainSample = X3_train[counter3]
                Y_trainSample = Y3_train[counter3]
                counter3 += 1
                counter3 %= len(X3_train)

            X_trainSample = torch.tensor(X_trainSample).type(torch.float)
            Y_trainSample = torch.tensor(Y_trainSample).type(torch.float)
            #print(type(X_trainSample))
            #print(type(Y_trainSample))
            #print(type(Y))
            train_loss = training(X_trainSample, Y_trainSample, model, loss_fn, optimizer)

             # Test model on random sample
            X_test = torch.tensor(X_test).type(torch.float)
            Y_test = torch.tensor(Y_test).type(torch.float)
            test_loss, accuracy = testing(X_test, Y_test, loss_fn, model)

             # Get results from current node testing
            test_losses.append(test_loss)
            accuracies.append(accuracy)
            if (i % 1000 == 0):
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

averaged_clustered_test_losses, averaged_clustered_accuracies = graphRandomWalkLearn(G, X1_train, X2_train, X3_train, Y1_train, Y2_train, Y3_train, X_test, Y_test, 
                                                                                     runs=RUNS, timeCount=ITERATIONS, plotResults=False, learning_rate=LEARNING_RATE)
#averaged_complete_test_losses, averaged_complete_accuracies = graphRandomWalkLearn(G2, X1_train, X2_train, X3_train, Y1_train, Y2_train, Y3_train, X_test, Y_test, 
#                                                                                     runs=RUNS, timeCount=ITERATIONS, plotResults=False, learning_rate=LEARNING_RATE)
#plotTestLosses(ITERATIONS, averaged_clustered_test_losses, xlabel="Iterations")
#plotAccuracies(ITERATIONS, averaged_clustered_accuracies, xlabel="Iterations")
saveArrToFile(averaged_clustered_accuracies, path="results/iris2/clustered.txt")
#saveArrToFile(averaged_complete_accuracies, path="results/complete.txt")

# Mix and train graph for various temperatures
n = 60_000
times = np.arange(n+1)
temperatures = [0.1, 1, 10, 100]
sampleTimeTensor = []
energyTensor = []
for i in range(len(temperatures)):
    # Perform mixing
    t = temperatures[i]
    print(f"Running with temperature: {t}")
    Gnew = mAryGraphGen(m=3, cluster_size=CLUSTER_SIZE, sparse_connections=10, p=0.3, path="graphData/mAryMixingGraph.csv", plotGraph=False)
    sampleTimes, energies, numGoodLinks, Gmixed = mAryGlauberDynamicsDataSwitch(3, Gnew, times, t, plot=False, samplingSize=100)

    # Post-mixing training/testing
    averaged_mixed_test_losses, averaged_mixed_accuracies = graphRandomWalkLearn(Gmixed, X1_train, X2_train, X3_train, Y1_train, Y2_train, Y3_train, X_test, Y_test, 
                                                                                        runs=RUNS, timeCount=ITERATIONS, plotResults=False, learning_rate=LEARNING_RATE)
    saveArrToFile(averaged_mixed_accuracies, path=f"results/iris2/mixed{i}.txt")

    # Save results
    sampleTimeTensor.append(sampleTimes)
    energyTensor.append(energies)

saveArrToFile(sampleTimeTensor, path="results/iris2/sampleTimes.txt")
saveArrToFile(energyTensor, path="results/iris2/energies.txt")

# Plot results
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
def plotCombinedAccuracies(iterations, centralized_accuracies, clustered_accuracies, mixed_accuracies, xlabel="Iterations"):
    x = np.arange(iterations)
    plt.title("Comparison of Learning Test Accuracies")
    plt.xlabel(xlabel)
    plt.ylabel("Test Accuracy")
    plt.ylim([0, 1.05])

    plt.plot(x, centralized_accuracies[:iterations], label="Centralized", color="royalblue")
    #plt.plot(x, complete_accuracies, label="Complete Graph")
    plt.plot(x, clustered_accuracies[:iterations], label="Random Walk Learning (Before Shuffling)", color="darkorange")
    plt.plot(x, mixed_accuracies[:iterations], label="Random Walk Learning (After Shuffling)", color="green")
    plt.legend()
    plt.show()
#plotCombinedAccuracies(ITERATIONS, averaged_centralized_accuracies, averaged_clustered_accuracies, averaged_mixed_accuracies)
