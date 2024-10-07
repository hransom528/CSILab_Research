# Harris Ransom
# PyG Distributed Data Store
# From PyG Distributed Training: https://pytorch-geometric.readthedocs.io/en/latest/tutorial/distributed_pyg.html

# Imports
import torch
from torch_geometric.distributed import LocalFeatureStore
from torch_geometric.distributed.event_loop import to_asyncio_future

feature_store = LocalFeatureStore(...)

async def get_node_features():
    # Create a `LocalFeatureStore` instance:

    # Retrieve node features for specific node IDs:
    node_id = torch.tensor([1])
    future = feature_store.lookup_features(node_id)

    return await to_asyncio_future(future)