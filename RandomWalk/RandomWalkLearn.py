# Harris Ransom
# Performs a random walk on a graph for ML classification
# 11/10/2024

# Imports
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
#from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
# TODO: Set up argparse

# Custom Imports
from Graph import Graph
from graphGen import graphGen
from TVDistance import tvDistance
from RandomWalk import MetropolisHastingsRandomWalk, plotTVDistances
from DataMixing import GlauberDynamicsDataSwitch, plotEnergy, plotDiffHist, plotGoodLinks
from MNISTData import loadMNISTData

# Imports sorted MNIST dataset
print("Loading MNIST data...")
SIZE = 100
TEST_SIZE = SIZE // 10
trainDataset, testDataset = loadMNISTData("MNIST_Data/trainset", "MNIST_Data/testset", train_size=SIZE, test_size=TEST_SIZE)
# Nodes 0-999 are 0, 1000-1999 are 1

# Set up the basic neural network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=28*28, out_features=512),
            nn.ReLU(),
            #nn.Linear(512, 512),
            #nn.ReLU(),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Hyperparameters
print("Setting up neural network...")
learning_rate = 1e-3
batch_size = 64
epochs = 5
#trainDataloader = DataLoader(trainDataset, batch_size=batch_size)
testDataloader = DataLoader(testDataset, batch_size=batch_size)
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device for training.")
model = NeuralNetwork().to(device)
loss_fn = nn.CrossEntropyLoss() # Initialize the loss function
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the model
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
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct, test_loss

# Set up unmixed graph (Erdos–Renyi model, p=0.1, 10 sparse connections)
print("Loading generated graph...")
loadPregen = False
if (loadPregen):
    UNMIXED_PATH = "graphData/generatedGraph.csv"
    UNMIXED_TYPES = UNMIXED_PATH + ".types"
    G = Graph.importCSV(UNMIXED_PATH, UNMIXED_TYPES) # TODO: Change this to importTypedCSV
else:
    G = graphGen(SIZE, int(SIZE*0.05), p=0.1, path="graphData/generatedGraph2.csv", plotGraph=False)

# Performs random walk and NN learning on graph
def randomWalkLearn(G, trainDataset, testDataloader, model, loss_fn, optimizer, max_time, step, batch_size=64):
    accuracies = []
    losses = []
    for n in range(10, max_time, step):  
        times = np.arange(n+1)
        print(f"Performing unmixed random walk (n = {n})...")
        nodesVisited, P, tvDistances = MetropolisHastingsRandomWalk(G, times)
        #plotTVDistances(times, tvDistances)

        # Load data from nodesVisited and train/test the model
        nodesVisited = list(set(nodesVisited))
        sampledImgs = []
        sampledLabels = []
        CLUSTER_SIZE = G.nodes // 2
        for i in nodesVisited:
            if (i > CLUSTER_SIZE):
                i = i - CLUSTER_SIZE
            if (G.getNodeType(i) == -1):
                sampledImgs.append(trainDataset[i][0])
                sampledLabels.append(trainDataset[i][1])
            else:
                sampledImgs.append(trainDataset[i+(CLUSTER_SIZE-1)][0])
                sampledLabels.append(trainDataset[i+(CLUSTER_SIZE-1)][1])
        sampledImgs = torch.tensor(np.array(sampledImgs))
        sampledLabels = torch.tensor(np.array(sampledLabels)).type(torch.long)
        sampledData = TensorDataset(sampledImgs, sampledLabels)

        # Train and test the model
        sampledDataloader = DataLoader(sampledData, batch_size=batch_size)
        train_loop(sampledDataloader, model, loss_fn, optimizer)
        correct, test_loss = test_loop(testDataloader, model, loss_fn)
        
        accuracies.append(correct) # Save results of testing
        losses.append(test_loss)
    return accuracies, losses

# Plots results of RWL
def plotRWLResults(max_time, step, accuracies, losses):
    plt.plot(np.arange(10, max_time, step), accuracies)
    plt.ylim((0, 1))
    plt.xlabel("Number of Nodes Visited")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Number of Nodes Visited")
    plt.show()

    plt.plot(np.arange(10, max_time, step), losses)
    plt.ylim((0, 1))
    plt.xlabel("Number of Nodes Visited")
    plt.ylabel("Loss")
    plt.title("Loss vs. Number of Nodes Visited")
    plt.show()

# Perform random walk of unmixed graph
print("Performing unmixed random walks...")
MAX_TIME = 20_000
STEP = 500
EPOCHS = 1
accuracyMatrix = []
lossMatrix = []
for i in range(EPOCHS):
    accuracies, losses = randomWalkLearn(G, trainDataset, testDataloader, model, loss_fn, optimizer, max_time=MAX_TIME, step=STEP, batch_size=batch_size)
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
# TODO: Get confidence interval of accuracies and losses

# Save results of unmixed random walk to text files
'''with open('graphData/unmixedAccuracies.txt', 'w') as outfile:
  outfile.write('\n'.join(str(i) for i in accuracies))
with open('graphData/unmixedLosses.txt', 'w') as outfile:
  outfile.write('\n'.join(str(i) for i in losses))'''

# Plot unmixed accuracy and losses vs. number of nodes visited
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
    n = 100_000
    times = np.arange(n+1)
    sampleTimes, energies, numGoodLinks, G = GlauberDynamicsDataSwitch(G, times, 0.05, plot=False, samplingSize=100)
    #plotEnergy(sampleTimes, energies)
    #plotDiffHist(G)
    #plotGoodLinks(sampleTimes, numGoodLinks)


# Perform random walk of mixed graph
print("Performing mixed random walk...")
accuracyMatrix = []
lossMatrix = []
for i in range(EPOCHS):
    accuracies, losses = randomWalkLearn(G, trainDataset, testDataloader, model, loss_fn, optimizer, max_time=MAX_TIME, step=STEP, batch_size=batch_size)
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