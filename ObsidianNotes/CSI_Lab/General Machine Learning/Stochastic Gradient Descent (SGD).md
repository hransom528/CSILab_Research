---

---
Stochastic Gradient Descent (SGD) is a variant of the [[Gradient Descent (GD)]] algorithm that takes advantage of random (stochastic) sampling of a subset of data in order to perform stochastic approximation of the gradient descent optimization. 

SGD is performed by:
1. Randomly shuffling a training dataset
2. Dividing the shuffled dataset into mini-batches (can be a uniform distribution or some weighted distribution - it's a tradeoff)
4. Computing an estimated gradient descent step on a subset of data using [[Backpropagation]]

SGD, in comparison to GD, has a lower [[Rate of Convergence]]  due to how it performs optimization on a subset of the data rather than on the entire dataset. 

SGD is particularly helpful for [[Federated Learning]] since multiple entities can collaboratively train a [[Machine Learning Model]] without needing the entire dataset, thus reducing communication overhead when compared to traditional [[Gradient Descent (GD)]]. 

**Key detail:** In the proof for SGD, it is shown that SGD does not necessarily decrease the value of the loss function at each step. However, the *expected* loss for each step is negative, which means that overall SGD will converge to a minimized loss function. 
