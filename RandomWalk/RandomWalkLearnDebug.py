# Harris Ransom
# Performs a random walk on a graph for ML classification
# 11/10/2024

# Imports
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
#from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
# import argparse TODO: Set up argparse

# Custom Imports
from Graph import Graph
from graphGen import graphGen
from TVDistance import tvDistance
from RandomWalk import MetropolisHastingsRandomWalk, plotTVDistances, plotRandomWalk
from DataMixing import GlauberDynamicsDataSwitch, plotEnergy, plotDiffHist, plotGoodLinks
from MNISTData import loadMNISTData

# Imports sorted MNIST dataset
print("Loading MNIST data...")
SIZE = 50
if (SIZE % 2 != 0):
    raise ValueError("Size is not evenly divisible by 2!")
TEST_SIZE = SIZE // 5
batch_size = 1
trainDataset, trainDataloader, testDataloader = loadMNISTData("MNIST_Data/trainset", "MNIST_Data/testset", train_size=SIZE, test_size=TEST_SIZE, batch_size=batch_size)

# Set up the basic neural network
LEARNING_RATE = 0.1
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=28*28, out_features=512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Alternatively: Logisitic Regression
class Logistic_Loss(torch.nn.Module):
    def __init__(self):
        super(Logistic_Loss, self).__init__()

    def forward(self, inputs, target):
        L = torch.log(1 + torch.exp(-target*inputs.t()))
        return torch.mean(L)
#model = nn.Linear(28*28, 1)
#criterion = Logistic_Loss()
#optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

# Hyperparameters
print("Setting up neural network...")
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device for training.")
model = NeuralNetwork().to(device)
#loss_fn = nn.CrossEntropyLoss() # Initialize the loss function
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

# Train the model
# See: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
'''
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            return loss, current
'''
            
# Test the model
def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = torch.squeeze(model(X))
            y = y.squeeze().type(torch.float)
            test_loss += loss_fn(pred, y).item()

            # Round predicition and y to ints
            pred = torch.round(pred)
            y = torch.round(y)

            correct += (pred == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct, test_loss

# Set up unmixed graph (Erdos–Renyi model, p=0.1, 10 sparse connections)
print("Loading generated graph...")
loadPregen = False
if (loadPregen):
    UNMIXED_PATH = "graphData/generatedGraph2.csv"
    UNMIXED_TYPES = UNMIXED_PATH + ".types"
    with open(UNMIXED_TYPES, "r") as typeFile:
        typeList = list(map(int, typeFile.readline().split(",")))
    G = Graph.importTypedCSV(UNMIXED_PATH, typeList)
else:
    G = graphGen(SIZE // 2, int(SIZE*0.05), p=0.15, path="graphData/generatedGraph2.csv", plotGraph=True)

# Performs random walk and NN learning on graph
def randomWalkLearn(G, trainDataset, testDataloader, model, loss_fn, optimizer, max_time, step):
    accuracies = []
    losses = []

    # Perform random walk
    times = np.arange(max_time+1)
    print(f"Performing unmixed random walk (n = {max_time})...")
    nodesVisited, P, tvDistances = MetropolisHastingsRandomWalk(G, times)
    #plotTVDistances(times, tvDistances)

    # Plot random walk path
    nodeTypes = []
    for i in nodesVisited:
        nodeType = G.getNodeType(i)
        nodeTypes.append(nodeType)
    #plotRandomWalk(times, nodeTypes)

    # Load data from nodesVisited and train/test the model
    CLUSTER_SIZE = G.nodes // 2
    for i in range(0, len(nodesVisited)):
        # Get typed data from node
        nodeVisited = nodesVisited[i]
        if (nodeTypes[i] == -1):
            sampledImg = trainDataset[nodeVisited][0]
            sampledLabel = trainDataset[nodeVisited][1]
        else:
            sampledImg = trainDataset[nodeVisited+(CLUSTER_SIZE-1)][0]
            sampledLabel = trainDataset[nodeVisited+(CLUSTER_SIZE-1)][1]

        # Convert label to +1/-1
        #sampledLabel = Tensor(sampledLabel).reshape([1])
        sampledLabel = Tensor([(2 * (sampledLabel - 0.5))])

        # Train the model off of new datapoint
        # See: https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
        #logits = model(sampledImg)
        #pred_prob = nn.Softmax(dim=1)(logits)
        #pred = pred_prob.argmax(1).squeeze().type(torch.float)
        model.train()
        optimizer.zero_grad()
        pred = (model(sampledImg))
        loss = loss_fn(pred, sampledLabel) # Loss function calculation

        # TODO: Reset weights between random walks
        loss.backward() # Backpropagation
        optimizer.step()
        
        # Test the model
        if (i % step == 0):
            correct, test_loss = test_loop(testDataloader, model, loss_fn)
            accuracies.append(correct) # Save results of testing
            losses.append(test_loss)
    plotRWLResults(max_time, step, accuracies, losses)
    return accuracies, losses, runningLossVals

# Plots results of RWL
def plotRWLResults(max_time, step, accuracies, losses):
    times = np.arange(0, max_time+1, step)
    plt.plot(times, accuracies)
    plt.ylim((0, 1))
    plt.xlabel("Number of Nodes Visited")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Number of Nodes Visited")
    plt.show()

    plt.plot(times, losses)
    plt.ylim((0, 1))
    plt.xlabel("Number of Nodes Visited")
    plt.ylabel("Loss")
    plt.title("Loss vs. Number of Nodes Visited")
    plt.show()

# Perform random walk of unmixed graph
print("Performing unmixed random walks...")
MAX_TIME = SIZE * 100
STEP = SIZE / 10
EPOCHS = 5
accuracyMatrix = []
lossMatrix = []
for i in range(EPOCHS):
    accuracies, losses, runningLossVals = randomWalkLearn(G, trainDataset, testDataloader, model, loss_fn, optimizer, max_time=MAX_TIME, step=STEP)
    accuracyMatrix.append(accuracies)
    lossMatrix.append(losses)
accuracyMatrix = np.array(accuracyMatrix)
print(accuracyMatrix.shape)
lossMatrix = np.array(lossMatrix)
if (EPOCHS == 1):
    averageAccuracies = accuracyMatrix
    averageLosses = lossMatrix
else:
    averageAccuracies = np.mean(accuracyMatrix, axis=1)
    averageLosses = np.mean(lossMatrix, axis=1)
# TODO: Get confidence interval of accuracies and losses

# Save results of unmixed random walk to text files
'''with open('graphData/unmixedAccuracies.txt', 'w') as outfile:
  outfile.write('\n'.join(str(i) for i in accuracies))
with open('graphData/unmixedLosses.txt', 'w') as outfile:
  outfile.write('\n'.join(str(i) for i in losses))'''

# Plot unmixed accuracy and losses vs. number of nodes visited
print(averageAccuracies)
plotRWLResults(MAX_TIME, STEP, averageAccuracies, averageLosses)

# Mix Graph
print("Load mixing graph...")
loadPremixed = False
if (loadPremixed):
    MIXED_PATH = "graphData/mixingGraph3.csv"
    MIXED_TYPES = MIXED_PATH + ".types"
    nodeTypes = []
    with open(MIXED_TYPES) as file:
        nodeTypes = [int(line.rstrip()) for line in file]
    G = Graph.importTypedCSV(MIXED_PATH, nodeTypes)
else:
    n = 10_000
    times = np.arange(n+1)
    sampleTimes, energies, numGoodLinks, G = GlauberDynamicsDataSwitch(G, times, 0.05, plot=False, samplingSize=1)
    #plotEnergy(sampleTimes, energies)
    #plotDiffHist(G)
    #plotGoodLinks(sampleTimes, numGoodLinks)

# Perform random walk of mixed graph
print("Performing mixed random walk...")
accuracyMatrix = []
lossMatrix = []
for i in range(EPOCHS):
    accuracies, losses = randomWalkLearn(G, trainDataset, testDataloader, model, loss_fn, optimizer, max_time=MAX_TIME, step=STEP)
    accuracyMatrix.append(accuracies)
    lossMatrix.append(losses)
accuracyMatrix = np.array(accuracyMatrix)
lossMatrix = np.array(lossMatrix)
if (EPOCHS == 1):
    averageAccuracies = accuracyMatrix
    averageLosses = lossMatrix
else:
    averageAccuracies = np.mean(accuracyMatrix, axis=1)
    averageLosses = np.mean(lossMatrix, axis=1)
plotRWLResults(MAX_TIME, STEP, averageAccuracies, averageLosses)

# Save results of mixed random walk to text files
with open('graphData/mixedAccuracies.txt', 'w') as outfile:
  outfile.write('\n'.join(str(i) for i in averageAccuracies))
with open('graphData/mixedLosses.txt', 'w') as outfile:
  outfile.write('\n'.join(str(i) for i in averageLosses))

# TODO: Dunder main