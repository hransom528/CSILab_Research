
Gradient Descent (GD) is the algorithm used to minimize the [[Loss Function]] of a [[Machine Learning Model]]. 

- **Recall**: The [[Gradient]] of a multivariate function f(x, y) is the vector field that maps the input to the vector associated with the path of steepest ascent. 
$$ \nabla f = <\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}>$$
- To *minimize* our function f(x, y), we want to travel in this direction of steepest descent by taking a step in that direction:

$$
(x_{n+1}, y_{n+1}) = (x_n,y_n) - \alpha \nabla f(x_n,y_n)
$$
- Starting from an initial guess $x_0$ , we can keep improving our optimization little by little until we find a local minimum. 
	- **Motivation:** If we have a huge number of variables (like we do in a [[Neural Network]]), directly calculating the gradient and setting it equal to 0 is impractical. 
- **Limitation**: GD only finds the *local* minimum of the function, not the global minimum. This means that it is entirely possible to converge to different local minimums depending on the random guess $x_0$ that you start with. 
	- This means that GD will never escape a valley/"ditch" greater than $\alpha$ .
	- There is no GD algorithm that can find the global minimum of a function (otherwise we wouldn't have this problem).
- **Note**: The step size $\alpha$ can be considered a [[Hyperparameter]] of your machine learning model. 
	- It is common to choose a larger $\alpha$ at the beginning (i.e. at smaller values of $n$) and then gradually shrinking $\alpha$ as the optimization progresses. This allows for both a faster initial [[Rate of Convergence]] and finer tuning later on. 

- We can actually optimize $\alpha$ to achieve a better [[Rate of Convergence]]. First, we define a new function $g(\alpha)$:
	$$ g(\alpha) = f((x_n, y_n) - \alpha \nabla f(x_n,y_n)) $$
	We can then perform *single-variable* optimization to get:
	$$ g'(\alpha) = 0 $$
	Repeating this process, we can effectively approach the max (or min) of the function $f(x, y)$ much faster while only having to perform optimization of a single variable. 