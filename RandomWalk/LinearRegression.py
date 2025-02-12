# Harris Ransom
# Random Walk Learning Debugging w/Linear Regression
# 2/6/2025

# Importing Libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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

# Define linear regression model
# TODO: Verify that this is accurate
# See: https://www.geeksforgeeks.org/linear-regression-using-pytorch/
# https://www.geeksforgeeks.org/classification-using-pytorch-linear-function/
class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(2, 2),
            torch.nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
LEARNING_RATE = 0.1
EPOCHS = 5000
#loss_fn = torch.nn.MSELoss(reduction='sum')
loss_fn = torch.nn.CrossEntropyLoss()
model = LinearRegression()
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
test_losses = []
accuracies = []
for i in range(EPOCHS):
    train_loss = training(X_train, Y_train, model, loss_fn, optimizer)
    test_loss, accuracy = testing(X_test, Y_test, model)
    test_losses.append(test_loss)
    accuracies.append(accuracy)
    if (i % 100 == 0):
        print(f"\nEpoch {i}: Training Loss = {train_loss.item()}")
        print(f"Epoch {i}: Testing Loss = {test_loss.item()}")
        print(f"Epoch {i}: Accuracy = {accuracy}")

# Plots test losses over time (epochs)
def plotTestLosses(epochs, test_losses):
    x = np.arange(epochs)
    plt.plot(x, test_losses)
    plt.title("Testing Losses over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Testing Loss (CrossEntropy)")
    plt.ylim([0, 1])
    plt.show()
plotTestLosses(EPOCHS, test_losses)

# Plots accuracies over time (epochs)
def plotAccuracies(epochs, accuracies):
    x = np.arange(epochs)
    plt.plot(x, accuracies)
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.ylim([0, 1.05])
    plt.show()
plotAccuracies(EPOCHS, accuracies)