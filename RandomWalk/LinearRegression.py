# Harris Ransom
# Random Walk Learning Debugging w/Linear Regression
# 2/6/2025

# Importing Libraries
import torch
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for consistency
SEED = 45
torch.manual_seed(SEED)
np.random.seed(SEED)

# Generate pseudo-random synthetic data parameters
SIZE = 100
START_X = -5
END_X = 5
RANGE_LEN = END_X - START_X
X = torch.arange(START_X, END_X, RANGE_LEN/SIZE).view(-1, 1)
m_1 = np.random.uniform(-10, 10)
m_2 = np.random.uniform(-10, 10)
b_1 = np.random.uniform(-10, 10)
b_2 = np.random.uniform(-10, 10)
print(f"No. of data points: {2*SIZE}")
print(f"Class 1: Y_1 = {m_1}(X) + {b_1}")
print(f"Class 2: Y_2 = {m_2}(X) + {b_2}")

# Generate noisy synthetic data
noise_mean_1 = np.random.uniform(-10, 10)
noise_mean_2 = np.random.uniform(-10, 10)
variance = np.random.uniform(5, 10)
print(f"\nClass 1 Noise Mean: {noise_mean_1}")
print(f"Class 2 Noise Mean: {noise_mean_2}")
print(f"Noise Variance: {variance}")
Y_1 = m_1*(X) + b_1 + np.random.normal(noise_mean_1, variance, X.size())
Y_2 = m_2*(X) + b_2 + np.random.normal(noise_mean_2, variance, X.size())

# Plot synthetic data
def plotGeneratedData(X, Y_1, Y_2):
    plt.figure()
    plt.scatter(X, Y_1, label='Class 1')
    plt.scatter(X, Y_2, label='Class 2')
    plt.title("Synthetic Data")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()
plotGeneratedData(X, Y_1, Y_2)

# Combine data into tensor objects
X = torch.cat((X, X))
Y = torch.cat((Y_1, Y_2))
#labels = torch.cat((torch.zeros(SIZE), torch.ones(SIZE))).reshape(-1, 1)

# Shuffles data
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
X, Y = unison_shuffled_copies(X, Y)
X = X.to(torch.float32)
Y = Y.to(torch.float32)

# Train-test split
SIZE = 2*SIZE # Accounts for both classes
PERCENT_TRAIN = 0.75
train_size = int(PERCENT_TRAIN * SIZE)
test_size = SIZE - train_size
X_train = X[:train_size]
Y_train = Y[:train_size]
X_test = X[train_size:]
Y_test = Y[train_size:]
print(f"\nTrain Size: {train_size}")
print(f"Test Size: {test_size}")
print(f"X_train: {X_train.size()}")
print(f"Y_train: {Y_train.size()}")
print(f"X_test: {X_test.size()}")
print(f"Y_test: {Y_test.size()}")

# Plot train-test split
def plotTrainTestSplit(X_train, Y_train, X_test, Y_test):
    plt.figure()
    plt.scatter(X_train, Y_train, label='Train Data')
    plt.scatter(X_test, Y_test, label='Test Data')
    plt.title("Train-Test Split")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()
plotTrainTestSplit(X_train, Y_train, X_test, Y_test)

# Define linear regression model
# TODO: Verify that this is accurate
# See: https://www.geeksforgeeks.org/linear-regression-using-pytorch/
class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
    
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
LEARNING_RATE = 0.01
EPOCHS = 20
loss_fn = torch.nn.MSELoss(reduction='sum')
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
    
    #with torch.no_grad():
    Y_pred = model(X)
    loss = loss_fn(Y_pred, Y_test)
    return loss

# Perform training and testing for multiple epochs
print("\nTraining and testing model...")
for i in range(EPOCHS):
    train_loss = training(X_train, Y_train, model, loss_fn, optimizer)
    print(f"Epoch {i}: Training Loss = {train_loss.item()}")
    test_loss = testing(X_test, Y_test, model)
    print(f"Epoch {i}: Testing Loss = {test_loss.item()}")

