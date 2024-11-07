import torch.nn as nn


class SDDOIALinear(nn.Module):

    def __init__(self, din, nconcept):
        super(SDDOIALinear, self).__init__()

        # set self hyperparameters
        self.din = din  # Input dimension
        self.nconcept = nconcept  # Number of "atoms"/concepts

        """
        encoding
        self.enc1: encoder for known concepts
        """
        self.bottleneck = nn.Sequential(
            nn.Linear(self.din, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.nconcept),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # resize
        p = x.view(x.size(0), -1)
        # return output
        return self.bottleneck(p)
