# Graph Data Mixing
Code, data, and figures for the federated machine learning and graph data mixing experiments. 

## Installing Dependencies
To install the required Python packages (if you don't already have them installed), you can run: `pip install -r requirements.txt`

## Included Files
- **DataMixing:** Functions that perform Glauber Dynamics data mixing on a given graph
- **FigureCreator:** Script that takes *LinearClassification* and *IrisData* output data from the *results/* directory and creates a set of specified figures
- **generateGraphAnimation:** Helper script to generate an animated GIF from a set of mixing figures
- **Graph:** Contains the base Graph class
- **graphGen:** Functions that generate various kinds of graphs (2-cluster, M-cluster, complete, E-R, D-regular, etc.) using the Graph class
- **graphTest:** Basic testing code for the Graph class
- **IrisData:** Script that performs training/testing of centralized and federated (mixed and unmixed) classification models using the Iris dataset 
- **LinearClassification:** Script that performs training/testing of centralized and federated classification models using synthetically generated data
- **MNISTData:** Helper code to get a filtered version of the MNIST dataset for federated learning experiments
- **RandomWalk:** Functions that perform a Metropolis-Hastings random walk on a given Graph
- **TVDistance:** Helper code to calculate the total variational (TV) distance between two probability distributions

## Example Usage
First, you can run the *LinearClassification* and *IrisData* scripts to generate data and figures for the synthetic and Iris datasets:
`python3 LinearClassification.py` and `python3 IrisData.py`

Before you run the two training/testing scripts, be sure to change all of the file paths where the results will be saved so that you don't overwrite any previous results. 

After you have gathered results (or if you want to use the data already in the *results/* directory), you can then run the *FigureCreator* script to generate a set of figures based off of the data files:
`python3 FigureCreator.py`

You can configure what data files you are using, and how you want to generate the figures, in the *FigureCreator* script. 