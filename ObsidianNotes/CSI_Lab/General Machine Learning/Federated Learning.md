Federated Learning (aka Collaborative Learning) is a sub-field of machine learning in which clients (aka nodes) collaboratively train a model while the data remains decentralized. 
**Note**: This means that the data in Federated Learning is *not* [[Independently and Identically Distributed (IID)]]. 


The [[Objective Function]] for a general Federated Learning architecture is:
$$ f(x_1,...,x_k) = \frac{1}{K} \sum_{i=1}^{K}f_i(x_i) $$ Where:
- $K$ is the number of nodes
- $x_i$ are the weights of the model as viewed by node $i$
- $f_i$ is node $i$'s local objective function, which describes how model weights $x_i$ conforms to node $i$'s local dataset

The goal of federated learning is to train a common model on all of the nodes' local datasets via:
- Optimizing the objective function $f(x_1,...,x_k)$
- Achieving [[Consensus]] on $x_i$, in that $x_1...x_k$ converge to some common $\vec{x}$ at the end of the training process. 