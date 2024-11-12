# Harris Ransom
# Performs a random walk on a graph for ML classification
# 11/10/2024

# Imports
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader

# Custom Imports
from Graph import Graph
from graphGen import graphGen
from TVDistance import tvDistance
from RandomWalk import MetropolisHastingsRandomWalk, plotTVDistances
from DataMixing import GlauberDynamicsDataSwitch, plotEnergy, plotDiffHist, plotGoodLinks
from MNISTData import loadMNISTData

# Imports sorted MNIST dataset
print("Loading MNIST data...")
trainDataset, testDataset = loadMNISTData("MNIST_Data/trainset", "MNIST_Data/testset", train_size=1000, test_size=100)
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

# Set up graph (Erdos–Renyi model, p=0.1, 10 sparse connections)
print("Loading generated graph...")
#G = graphGen(1000, 50, p=0.1, path="graphData/generatedGraph.csv", plotGraph=False)
G = Graph.importCSV("graphData/generatedGraph.csv")

# Perform random walk of unmixed graph
accuracies = []
for n in range(10, 50000, 100):  
    times = np.arange(n+1)
    print("Performing unmixed random walk...")
    nodesVisited, P, tvDistances = MetropolisHastingsRandomWalk(G, times)
    #plotTVDistances(times, tvDistances)

    # TODO: Load data from nodesVisited and train/test the model
    nodesVisited = list(set(nodesVisited))
    sampledImgs = []
    sampledLabels = []
    for i in nodesVisited:
        if (i > 1000):
            i = i - 1000
        if (G.getNodeType(i) == -1):
            sampledImgs.append(trainDataset[i][0])
            sampledLabels.append(trainDataset[i][1])
        else:
            sampledImgs.append(trainDataset[i+999][0])
            sampledLabels.append(trainDataset[i+999][1])
    sampledImgs = torch.tensor(np.array(sampledImgs))
    sampledLabels = torch.tensor(np.array(sampledLabels))
    sampledData = TensorDataset(sampledImgs, sampledLabels)

    # Train and test the model
    sampledDataloader = DataLoader(sampledData, batch_size=batch_size)
    train_loop(sampledDataloader, model, loss_fn, optimizer)
    correct, test_loss = test_loop(testDataloader, model, loss_fn)
    accuracies.append(correct) # Save results of testing

# Plot unmixed accuracy vs. number of nodes visited
plt.plot(np.arange(10, 50000, 100), accuracies)
plt.xlabel("Number of Nodes Visited")
plt.ylabel("Accuracy")
plt.title("Unmixed Accuracy vs. Number of Nodes Visited")

# Mix Graph
n = 100000
times = np.arange(n+1)
print("Mixing graph...")
#energies, numGoodLinks = GlauberDynamicsDataSwitch(G, times, 0.1, plot=False)
#plotEnergy(times, energies)
#plotDiffHist(G)
#plotGoodLinks(times, numGoodLinks)
# TODO: plot % good links

# Perform random walk of mixed graph
print("Performing mixed random walk...")
sd = G.getStationaryDistribution()
#nodesVisited, P, tvDistances = MetropolisHastingsRandomWalk(G, sd, times)
#plotTVDistances(times, tvDistances)