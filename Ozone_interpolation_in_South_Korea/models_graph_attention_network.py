"""
A pytorch geometric graph attention network.
"""

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    """
    A Graph Convolutional Net with attention
    """
    def __init__(self, n_inputs):
        super(GAT, self).__init__()
        self.linear1 = nn.Linear(n_inputs, 20)
        self.linear2 = nn.Linear(20, 5)
        self.conv = GATConv(5, 1, edge_dim=1)

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.conv(x, edge_index, edge_attr,
                      return_attention_weights=False)
        return x[0].view(-1)
