import torch


from torch import Tensor
from torch import nn


class EncoderConv64(nn.Module):
    """
    Convolutional encoder used in beta-VAE paper for the chairs data.
    Based on row 4-6 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
    Concepts with a Constrained Variational Framework"
    (https://openreview.net/forum?id=Sy2fzU9gl)

    """

    def __init__(self, x_shape=(3, 64, 64), z_size=6, z_multiplier=1):
        # checks
        (C, H, W) = x_shape
        assert (H, W) == (64, 64), "This model only works with image size 64x64."
        super().__init__()

        self.z_size = z_size
        self.z_total = z_size * z_multiplier

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=C, out_channels=32, kernel_size=4, stride=2, padding=2
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=2
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=2
            ),
            nn.ReLU(
                inplace=True
            ),  # This was reverted to kernel size 4x4 from 2x2, to match beta-vae paper
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=2
            ),
            nn.ReLU(
                inplace=True
            ),  # This was reverted to kernel size 4x4 from 2x2, to match beta-vae paper
            nn.Flatten(),
            nn.Linear(in_features=1600, out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=256, out_features=self.z_total
            ),  # we combine the two networks in the reference implementation and use torch.chunk(2, dim=-1) to get mu & logvar
        )

    def forward(self, x) -> Tensor:
        return torch.split(self.model(x), self.z_size, -1)


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


class DecoderConvBOIA(nn.Module):
    """
    Convolutional decoder used in beta-VAE paper for the chairs data.
    Based on row 3 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
    Concepts with a Constrained Variational Framework"
    (https://openreview.net/forum?id=Sy2fzU9gl)
    """

    def __init__(
        self,
        x_shape=(3, 387, 469),
        z_size=6,
        z_multiplier=1,
    ):
        (C, H, W) = x_shape
        assert (H, W) == (387, 469), "This model only works with image size 387x469."
        super().__init__()

        self.z_size = z_size

        self.model = nn.Sequential(
            nn.Linear(in_features=self.z_size, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=64 * 4 * 4),
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


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
