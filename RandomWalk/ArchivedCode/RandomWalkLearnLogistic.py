# Harris Ransom
# Performs a random walk on a graph for ML classification
# 2/4/2025

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
SIZE = 50 # TODO: Increase size or reduce features
if (SIZE % 2 != 0):
    raise ValueError("Size is not evenly divisible by 2!")
TEST_SIZE = SIZE // 5
batch_size = 1
trainDataset, trainDataloader, testDataloader = loadMNISTData("MNIST_Data/trainset", "MNIST_Data/testset", train_size=SIZE, test_size=TEST_SIZE, batch_size=batch_size)

# Sets up Logisitic Regression
LEARNING_RATE = 0.1
class Logistic_Loss(torch.nn.Module):
    def __init__(self):
        super(Logistic_Loss, self).__init__()

    def forward(self, inputs, target):
        L = torch.log(1 + torch.exp(-target*inputs.t()))
        return torch.mean(L)
model = nn.Linear(28*28, 1)
loss_fn = Logistic_Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

# Train the model
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
     
    total_loss = 0
    for i, (image, label) in enumerate(dataloader):
        image = image.reshape([-1, 28*28])
        label = 2 * (label.float() - 0.5)

         # Forward pass
        optimizer.zero_grad()
        output = model(image)
        loss = loss_fn(output, label)

        # Backward pass + Optimize
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
     
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
    total = 0
    correct = 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.reshape([-1, 28*28])
            test_pred = model(X)
            prediction = test_pred.data.sign()
            print(prediction)

            test_loss += loss_fn(prediction, y).item()

            correct += (prediction.view(-1).long() == y).sum()
            #correct += (pred == y).type(torch.float).sum().item()

            total += X.shape[0]
    print(correct)
    test_loss /= num_batches
    correct = correct / size
    
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct, test_loss

# Perform training/testing of model
EPOCHS = 5
for i in range(EPOCHS):
    train_loop(trainDataloader, model, loss_fn, optimizer)
    test_loop(testDataloader, model, loss_fn)