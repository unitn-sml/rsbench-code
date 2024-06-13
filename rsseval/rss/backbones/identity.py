import torch


from torch import Tensor
from torch import nn


class Identity(nn.Module):
    def __init__(self, z_dim, z_multiplier):
        # checks
        super().__init__()

        self.z_size = z_dim
        self.z_total = z_dim * z_multiplier

    def forward(self, x) -> Tensor:
        return x
