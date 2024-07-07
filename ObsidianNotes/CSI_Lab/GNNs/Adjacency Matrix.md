A convenient way to represent [[Graphs]] is through an **adjacency matrix** $\mathbf{A} \in \mathbb{R}^{|V|\times|V|}$ .

In an adjacency matrix, the nodes are ordered such that every node indexes a particular row and column within the matrix. We can then represent the prescence or absence of [[Edge]]s as entries in this matrix:
$\textbf{A}[u, v] = 1$ if $(u, v) \in \textbf{E}$ and $\textbf{A}[u, v] = 0$ otherwise. 

**Note:** If the graph is an [[Undirected Graph]] then the adjacency matrix $\mathbf{A}$ will be symmetric, otherwise if the graph is a [[Directed Graph]] then $\mathbf{A}$ will not necessarily be symmetric. 

**Note:** If the graph is a [[Weighted Graph]], then the entries in the adjacency matrix associated with the [[Edge]]s can be arbitrary real values rather than $\{0, 1\}$. 
