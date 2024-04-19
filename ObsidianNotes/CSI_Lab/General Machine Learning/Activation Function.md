An Activation Function in the context of a [[Neural Network]] is a function that calculates the output of an individual node based on its individual inputs and their weights (along with some bias value). 

Common activation functions can include:
- **Linear Activation Function:** $\phi(v) = a+v'b$
- [[ReLU Activation Function]]
- [[Sigmoid Function]] 


For a given node in a [[Neural Network]], an example activation function calculation using the [[Sigmoid Function]] would look like:
$$
a_j^{(L)}(\vec{w}, \vec{a}, b) = \sigma(z_j^{(L)}) = \sigma(\sum_{i=1}^{n}(w_{jk}^{(L)} a_k^{(L-1)}) + b_j^{(L)})
$$
Where:
- $a_j^{(L)}(\vec{w}, \vec{a}, b)$ is the result of the activation function calculation on the jth node of the Lth layer.  
- $\sigma(...)$ is an exemplary Sigmoid activation function
- $z^{L}$ is the weighted sum of the activations and bias
- $\vec{w}$ is the weight vector associated with the connections to the previous layer
- $\vec{a}$ are the values of the nodes in the previous layer
- $b$ is the bias of the current node