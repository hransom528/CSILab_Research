#!/bin/sh
# Harris Ransom
# 07/07/2024
# This script installs PyG on a Linux system (with Cuda)

# Install PyTorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install PyG
pip install torch_geometric

# Optional PyG libraries
pip install pyg-lib -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
pip install torch_scatter torch_sparse torch_cluster 
pip install torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
