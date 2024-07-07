**Heterogeneous graphs** are a subset of [[Graphs]] in which nodes are also imbued with *types*.
This means that we can partition the set of nodes into disjoint sets such that:
$$
\mathbf{V} = \mathbf{V_1} \cup \mathbf{V_2} \cup ...\cup \mathbf{V_k}
$$
where $\mathbf{V_i} \cap \mathbf{V_j} = \varnothing, \forall i \neq j$.

**Note:** Edges in heterogeneous graphs generally satisfy constraints according to the node types, most commonly the constraint that certain edges only connect nodes of certain types. 

**Note:** [[Multipartite Graphs]] are a well-known special case of heterogeneous graphs where edges can only connect nodes that have different types.

![Heterogeneous graph schematic, where green indicates word nodes, blue ...](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.researchgate.net%2Fpublication%2F370701676%2Ffigure%2Ffig1%2FAS%3A11431281157803503%401683909585990%2FHeterogeneous-graph-schematic-where-green-indicates-word-nodes-blue-indicates-sentence.jpg&f=1&nofb=1&ipt=403cbcaddeaf73b12ed23ec1f0ea5ab06f69cd5d2a039a4a4c4b6a88ef1f10ac&ipo=images)