We can extend our notation for an [[Edge]] to include different *types* of edges:
$$
(u, \tau, v) \in \mathbf{E}
$$
We can then define one [[Adjacency Matrix]] $\mathbf{A}_\tau$ per edge type. Such graphs are referred to as **multi-relational graphs** and can be summarized by an [[Adjacency Tensor]]:
$$
\mathbf{A} \in \mathbb{R}^{|V| \times |R| \times |V|},
$$
such that $R$ is a set of relations. 

Two important subsets of multi-relational graphs are [[Heterogeneous Graphs]] and [[Multiplex Graphs]].
