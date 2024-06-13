import torch


from torch import Tensor
from torch import nn


class SimpleMLP(nn.Module):
    def __init__(self, z_dim=6, z_multiplier=1):
        # checks
        super().__init__()

        self.z_size = z_dim
        self.z_total = z_dim * z_multiplier

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=128),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=128, out_features=self.z_total
            ),  # we combine the two networks in the reference implementation and use torch.chunk(2, dim=-1) to get mu & logvar
        )

        # self.model = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(in_features=2048, out_features=512),
        #     nn.Dropout(p=0.5),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(in_features=512, out_features=128),
        #     nn.Dropout(p=0.5),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(
        #         in_features=128, out_features=self.z_total
        #     ),  # we combine the two networks in the reference implementation and use torch.chunk(2, dim=-1) to get mu & logvar
        # )

    def forward(self, x) -> Tensor:
        return torch.split(self.model(x), self.z_size, -1)
