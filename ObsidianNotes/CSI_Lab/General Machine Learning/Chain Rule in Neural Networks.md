When performing [[Backpropagation]] as part of training a [[Neural Network]], the calculus involves using a specific form of the Chain Rule applied to an [[Activation Function]] and a [[Loss Function]].


The Chain Rule in the context of Neural Networks can be defined as:
$$
\frac{\partial C_0}{\partial w_{jk}^{(L)}} = \frac{\partial z_j^{(L)}}{\partial w_{jk}^{(L)}} \frac{\partial a_j^{(L)}}{\partial z_j^{(L)}} \frac{\partial C_0}{\partial a_j^{(L)}}
$$

Where:
- $C_0$ is the cost function of the current network
	- $C_0(...) = (a^{(L)}-y)^2$ where y is the desired output
- $a_k^{(L)}$ is the result of the activation function calculation on the kth node of the Lth layer.  
- $z^{L}$ is the weighted sum of the activations and bias of layer L
	- $z_j^{(L)} = w_{jk}^{(L)}a_k^{(L-1)} + b_j^{(L)}$
- $\vec{w}^{(L)}$ is the weight vector of the current layer L associated with the connections to the previous layer
- $j$ is the index of the node in the current layer $L$
- $k$ is the index of the node in the previous layer $L-1$

### Deriving the Result:

For a given cost function $C_0(...) = \sum_{j=0}^{n_L - 1} (a_j^{(L)}-y_j)^2$:
$$
\frac{\partial C_0}{\partial a_j^{(L)}} = 2(a_j^{(L)} - y_j)
$$

For a given activation function $a^{(L)}$ and weighted sum $z_j^{(L)}$:
$$
\frac{\partial a_j^{(L)})}{\partial z_j^{(L)}} = \sigma'(z_j^{(L)})
$$
For a given weighted sum $z_j^{(L)}$ and weight vector $w_{jk}^{(L)}$:
$$
\frac{\partial z_j^{(L)}}{\partial w_{jk}^{(L)}} = a_k^{(L-1)}
$$

Therefore, the result of applying the Chain Rule to a [[Neural Network]] is:
$$
\frac{\partial C_0}{\partial w_{jk}^{(L)}} = \frac{\partial z_j^{(L)}}{\partial w_{jk}^{(L)}} \frac{\partial a_j^{(L)}}{\partial z_j^{(L)}} \frac{\partial C_0}{\partial a_j^{(L)}} = a_k^{(L-1)} \sigma'(z_j^{(L)}) 2(a_j^{(L)}-y_j)
$$
and 
$$
\frac{\partial C_0}{\partial b_j^{(L)}} = \frac{\partial z_j^{(L)}}{\partial b_j^{(L)}} \frac{\partial a_j^{(L)}}{\partial z_j^{(L)}} \frac{\partial C_0}{\partial a_j^{(L)}} = \sigma'(z_j^{(L)}) 2(a_j^{(L)}-y_j)
$$
and
$$
\frac{\partial C_0}{\partial a_k^{(L-1)}} = \sum_{j=0}^{n_L-1}\frac{\partial z_j^{(L)}}{\partial a_k^{(L-1)}} \frac{\partial a_j^{(L)}}{\partial z_j^{(L)}} \frac{\partial C_0}{\partial a_j^{(L)}} = w_{jk}^{(L)} \sigma'(z_j^{(L)}) 2(a_j^{(L)}-y_j)
$$

**Note:** We can represent each individual weight $w_{jk}$ in a **weight matrix**. 


Source: 3B1B Backpropagation Calculus