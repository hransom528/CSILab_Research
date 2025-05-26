# Figure Creator
# Creates figures from saved simulation data
# 5/24/2025

# Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Custom imports
from Graph import Graph
from graphGen import graphGen, mAryGraphGen, mAryCompleteGraphGen
from TVDistance import tvDistance
from DataMixing import GlauberDynamicsDataSwitch, mAryGlauberDynamicsDataSwitch, plotEnergy, plotDiffHist, plotGoodLinks, plotTVHist

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

    #plt.plot(x, centralized_accuracies[:iterations], label="Centralized", color="royalblue")
    #plt.plot(x, complete_accuracies, label="Complete Graph")
    plt.plot(x, clustered_accuracies[:iterations], label="Clustered Erdos-Renyi (Unmixed)", color="darkorange")
    plt.plot(x, mixed_accuracies[:iterations], label="Clustered Erdos-Renyi (Mixed)", color="green")
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
#plotCombinedAccuracies(3000, synth_centralized_accuracies, synth_clustered_accuracies, synth_mixed_accuracies, xlabel="Iterations")
#plotClusteredAccuracy(50000, synth_clustered_accuracies, xlabel="Iterations")

# Iris Data Figures
iris_centralized_accuracies = loadArrFromFile(path="results/iris/centralized.txt")
iris_clustered_accuracies = loadArrFromFile(path="results/iris/clustered.txt")
iris_mixed_accuracies = loadArrFromFile(path="results/iris/mixed.txt")
#plotCombinedAccuracies(4000, iris_centralized_accuracies, iris_clustered_accuracies, iris_mixed_accuracies, xlabel="Iterations")
#plotClusteredAccuracy(50000, iris_clustered_accuracies, xlabel="Iterations")

# Synthetic Data Mixing Figures
'''
TRAIN_SIZE = 75
G = graphGen(cluster_size=TRAIN_SIZE, sparse_connections=15, p=0.1, path="./graphData/LinearRegressionGraph.csv", plotGraph=True)

n = 75_000
times = np.arange(n+1)
sampleTimes, energies, numGoodLinks, Gmixed = GlauberDynamicsDataSwitch(G, times, 0.1, plot=False, samplingSize=100)
plotEnergy(sampleTimes, energies)
plotDiffHist(Gmixed)
Gmixed.plot_typed_graph()
'''

# Iris Mixing Figures
'''
CLUSTER_SIZE = 40
G = mAryGraphGen(m=3, cluster_size=CLUSTER_SIZE, sparse_connections=5, p=0.3, path="graphData/generatedMAryClusteredGraph.csv", plotGraph=False)
n = 60_000
times = np.arange(n+1)
sampleTimes, energies, numGoodLinks, Gmixed = mAryGlauberDynamicsDataSwitch(3, G, times, 10, plot=False, samplingSize=100)
plotEnergy(sampleTimes, energies)
'''