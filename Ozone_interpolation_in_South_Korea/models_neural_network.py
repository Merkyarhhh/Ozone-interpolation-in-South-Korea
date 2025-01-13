"""
A simple Neural Net
"""

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F


class NN(torch.nn.Module):
    """
    A simple Neural Net
    """
    def __init__(self, n_inputs):
        super().__init__()
        self.linear1 = nn.Linear(n_inputs, 20)
        self.linear2 = nn.Linear(20, 5)
        self.linear3 = nn.Linear(5, 1)

    def forward(self, x, _, __):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x.view(-1)
