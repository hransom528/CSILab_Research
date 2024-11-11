# Harris Ransom
# MNIST Data Selection and Classification

# Imports
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader

# Define constants/parameters
trainsetPath = 'MNIST_Data/trainset'
testsetPath = 'MNIST_Data/testset'

# Load MNIST data and filter/sort for 0 and 1
def loadMNISTData(trainsetPath, testsetPath):
    # Load MNIST data
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])
    trainset = datasets.MNIST(trainsetPath, download=True, train=True, transform=transform)
    testset = datasets.MNIST(testsetPath, download=True, train=False, transform=transform)
    #trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    #testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    # Separate data and labels for train and test sets
    trainImgs = []
    trainLabels = []
    for i in range(0, len(trainset)):
        img, label = trainset[i]
        trainImgs.append(img)
        trainLabels.append(label)

    testImgs = []
    testLabels = []
    for i in range(0, len(testset)):
        img, label = testset[i]
        testImgs.append(img)
        testLabels.append(label)

    # Filter dataset for 0 and 1
    trainSet = [trainImgs[key] for (key, label) in enumerate(trainLabels) if int(label) == 0 or int(label) == 1]
    trainLabels = [label for label in trainLabels if int(label) == 0 or int(label) == 1]
    testSet = [testImgs[key] for (key, label) in enumerate(testLabels) if int(label) == 0 or int(label) == 1]
    testLabels = [label for label in testLabels if int(label) == 0 or int(label) == 1]

    # Only select 2000 images for training and testing
    # 1000 of each class
    selectedTrainImgs = []
    selectedTrainLabels = []
    zeroCount = 0
    oneCount = 0
    TRAIN_SIZE = 1000
    for i in range(0, len(trainLabels)):
        if zeroCount < TRAIN_SIZE and trainLabels[i] == 0:
            selectedTrainImgs.append(trainSet[i])
            selectedTrainLabels.append(trainLabels[i])
            zeroCount += 1
        elif oneCount < TRAIN_SIZE and trainLabels[i] == 1:
            selectedTrainImgs.append(trainSet[i])
            selectedTrainLabels.append(trainLabels[i])
            oneCount += 1
        if zeroCount == TRAIN_SIZE and oneCount == TRAIN_SIZE:
            break

    selectedTestImgs = []
    selectedTestLabels = []
    zeroCount = 0
    oneCount = 0
    TEST_SIZE = 100
    for i in range(0, len(testLabels)):
        if zeroCount < TEST_SIZE and testLabels[i] == 0:
            selectedTestImgs.append(testSet[i])
            selectedTestLabels.append(testLabels[i])
            zeroCount += 1
        elif oneCount < TEST_SIZE and testLabels[i] == 1:
            selectedTestImgs.append(testSet[i])
            selectedTestLabels.append(testLabels[i])
            oneCount += 1
        if zeroCount == TEST_SIZE and oneCount == TEST_SIZE:
            break

    # Sort selected data into 0 and 1 groups
    argSortedTrainLabels = np.argsort(selectedTrainLabels)
    argSortedTestLabels = np.argsort(selectedTestLabels)

    sortedTrainImgs = torch.tensor(np.array(selectedTrainImgs)[argSortedTrainLabels])
    sortedTrainLabels = torch.tensor(np.array(selectedTrainLabels)[argSortedTrainLabels])
    sortedTestImgs = torch.tensor(np.array(selectedTestImgs)[argSortedTestLabels])
    sortedTestLabels = torch.tensor(np.array(selectedTestLabels)[argSortedTestLabels])

    trainDataset = TensorDataset(sortedTrainImgs, sortedTrainLabels)
    testDataset = TensorDataset(sortedTestImgs, sortedTestLabels)

    # Export data
    return trainDataset, testDataset

# MAIN
if __name__ == "__main__":
    trainDataset, testDataset = loadMNISTData(trainsetPath, testsetPath)
    trainDataloader = DataLoader(trainDataset)
    testDataloader = DataLoader(testDataset)

    # Display image and label.
    train_features, train_labels = next(iter(trainDataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")
    