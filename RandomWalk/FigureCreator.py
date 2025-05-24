# Figure Creator
# Creates figures from saved simulation data
# 5/24/2025

# Imports
import numpy as np
import matplotlib.pyplot as plt

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

# Synthetic Data Figures
synth_centralized_accuracies = loadArrFromFile(path="results/synth/centralized.txt")
synth_clustered_accuracies = loadArrFromFile(path="results/synth/clustered.txt")
synth_mixed_accuracies = loadArrFromFile(path="results/synth/mixed.txt")
def plotCombinedAccuracies(iterations, centralized_accuracies, clustered_accuracies, mixed_accuracies, xlabel="Iterations"):
    x = np.arange(iterations)
    plt.title("Comparison of Learning Test Accuracies")
    plt.xlabel(xlabel)
    plt.ylabel("Test Accuracy")
    plt.ylim([0, 1.05])

    plt.plot(x, centralized_accuracies[:iterations], label="Centralized")
    #plt.plot(x, complete_accuracies, label="Complete Graph")
    plt.plot(x, clustered_accuracies[:iterations], label="Clustered Erdos-Renyi (Unmixed)")
    plt.plot(x, mixed_accuracies[:iterations], label="Clustered Erdos-Renyi (Mixed)")
    plt.legend()
    plt.show()
def plotClusteredAccuracy(iterations, accuracies, xlabel="Iterations"):
    x = np.arange(iterations)
    plt.plot(x, accuracies, color="darkorange")
    plt.title("Averaged Clustered Accuracies")
    plt.xlabel(xlabel)
    plt.ylabel("Accuracy")
    plt.ylim([0, 1.05])
    plt.show()
plotCombinedAccuracies(5000, synth_centralized_accuracies, synth_clustered_accuracies, synth_mixed_accuracies, xlabel="Iterations")
plotClusteredAccuracy(50000, synth_clustered_accuracies, xlabel="Iterations")

# Iris Data Figures
iris_centralized_accuracies = loadArrFromFile(path="results/iris/centralized.txt")
iris_clustered_accuracies = loadArrFromFile(path="results/iris/clustered.txt")
iris_mixed_accuracies = loadArrFromFile(path="results/iris/mixed.txt")
plotCombinedAccuracies(5000, iris_centralized_accuracies, iris_clustered_accuracies, iris_mixed_accuracies, xlabel="Iterations")
plotClusteredAccuracy(50000, iris_clustered_accuracies, xlabel="Iterations")