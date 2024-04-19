A **loss function** (a.k.a. cost or error function) that maps events/values of one or more variables onto a real number which represents some "cost" associated with those events/values. 

A popular loss function is the [[Residual Sum of Squares (RSS)]]. 

In a neural network, [[Gradient Descent (GD)]] attempts to minimizes the average of the loss function of the model over the *entire* set of training data. 

For example, if we use RSS as our cost function for an individual piece of test or training data, we can then find the overall average cost across the entire training set as:
$$
\bar{RSS} = \frac{1}{N} \sum_{k=1}^{N} RSS = \frac{1}{N} \sum_{k=1}^{N} \sum_{i=1}^{n}(f(x_i)-\hat{f}(x_i))]
$$

Put more generally:
$$
C_0(...) = (a^{(L)}-y)^2
$$
where:
- $C_0$ is the cost of the current network
- $a^{(L)}$ is the activation of the current node
- $y$ is the desired output of the node