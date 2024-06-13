import torchvision
from torchvision.models import (
    resnet18,
    resnet50,
    resnet101,
    ResNet101_Weights,
    ResNet50_Weights,
    ResNet18_Weights,
)
import torch.nn as nn
import torch
from torch import Tensor


class ResNetEncoder(nn.Module):
    """
    Convolutional encoder used in beta-VAE paper for the chairs data.
    Based on row 4-6 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
    Concepts with a Constrained Variational Framework"
    (https://openreview.net/forum?id=Sy2fzU9gl)

    """

    def __init__(self, z_dim=18, z_multiplier=1, c_dim=10):
        super().__init__()

        self.pretrained = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        features = nn.ModuleList(self.pretrained.children())[:-1]

        # concept dim
        self.c_dim = c_dim

        # set the ResNet18 backbone as feature extractor
        self.encoder = nn.Sequential(*features)

        # Freeze all layers
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.bottleneck = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, c_dim),
            nn.Sigmoid(),
        )

    def forward(self, x) -> Tensor:
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        return self.bottleneck(x)


class DecoderConv64(nn.Module):
    """
    Convolutional decoder used in beta-VAE paper for the chairs data.
    Based on row 3 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
    Concepts with a Constrained Variational Framework"
    (https://openreview.net/forum?id=Sy2fzU9gl)
    """

    def __init__(
        self,
        x_shape=(3, 64, 64),
        z_size=6,
        z_multiplier=1,
    ):
        (C, H, W) = x_shape
        assert (H, W) == (64, 64), "This model only works with image size 64x64."
        super().__init__()

        self.z_size = z_size

        self.model = nn.Sequential(
            nn.Linear(in_features=self.z_size, out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Unflatten(dim=1, unflattened_size=[64, 4, 4]),
            nn.ConvTranspose2d(
                in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=32, out_channels=C, kernel_size=4, stride=2, padding=1
            ),
        )

    def forward(self, z) -> Tensor:
        return self.model(z)
