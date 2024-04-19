The [[Scaffold]] algorithm tries to correct for [[Client Drift]] by estimating the update direction for the server model $c$ and the update direction for each client $c_i$. Each of these estimates are referred to as **control variates**. 
The estimate of client drift used to correct the local update can then be calculated as:
$$
\textbf{error} = (c-c_i)
$$
