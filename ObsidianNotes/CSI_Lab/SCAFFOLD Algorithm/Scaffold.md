The SCAFFOLD (Stochastic Controlled Averaging for [[Federated Learning]]) paper seeks to:
1. Obtain tight convergence rates for [[Federated Averaging (FedAvg)]] and prove that it suffers from [[Client Drift]] when the data is non-[[Independently and Identically Distributed (IID)]]. 
2. Propose a new algorithm (SCAFFOLD) that uses [[Control Variates]] to correct for [[Client Drift]] in its local updates. 

### Scaffold Algorithm:
The SCAFFOLD Algorithm has 3 main steps:
1. Local updates to the client model
2. Local updates to the client control variate
3. Aggregating the updates

![[ScaffoldNotation.png]]

![[ScaffoldAlgorithm.png]]

To update the local control variate $c_i$, SCAFFOLD provides two options:
1. $g_i(x)$: Make an additional pass over the local data to compute the gradient at the server model $x$. 
2. $c_i-c+\frac{1}{K\eta_l}(x-y_i)$: Re-use the previously computed gradients to update the control variate. 
Option 2 is cheaper to compute and usually suffices, but Option 1 can be more stable. 


**Note:** The clients in SCAFFOLD are stateful and retain the value of their local control variate across multiple rounds.
**Note:** If the local control variates $c_i$ are set to 0, then SCAFFOLD becomes equivalent to [[Federated Averaging (FedAvg)]]. 