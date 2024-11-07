import copy
import torch.nn
from torch import nn


class Flatten(nn.Module):
    def forward(self, input):
        return torch.flatten(input, start_dim=1)


class UnFlatten(nn.Module):
    def forward(self, input, hidden_channels, dim):
        return input.reshape(input.size(0), hidden_channels, dim[0], dim[1])


class TripleCNNEncoder(nn.Module):
    NAME = "TripleCNNEncoder"

    def __init__(
        self,
        img_channels=3,
        hidden_channels=32,
        latent_dim=8,
        label_dim=20,
        dropout=0.5,
        img_concept_size=28,
    ):
        super(TripleCNNEncoder, self).__init__()

        self.img_concept_size = img_concept_size
        self.channels = 3
        self.img_channels = img_channels
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim
        self.label_dim = label_dim
        self.unflatten_dim = (3, 7)

        self.backbone = torch.nn.Sequential(
            nn.Conv2d(
                in_channels=self.img_channels,
                out_channels=self.hidden_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            torch.nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels * 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),  # 2*hidden_channels x 7 x 14
            torch.nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(
                in_channels=self.hidden_channels * 2,
                out_channels=self.hidden_channels * 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),  # 4*hidden_channels x 2 x 7
            torch.nn.ReLU(),
            Flatten(),
            nn.Linear(
                in_features=int(
                    4
                    * self.hidden_channels
                    * self.unflatten_dim[0]
                    * self.unflatten_dim[1]
                    * (3 / 7)
                ),
                out_features=self.latent_dim,
            ),
        )
        # self.backbone2= copy.deepcopy(self.backbone1)

    def forward(self, x):
        # MNISTPairsEncoder block 1
        # x = x.view(-1,self.channels,self.img_concept_size,self.img_concept_size*2)

        # x = x.view(-1, self.channels, self.img_concept_size, self.img_concept_size*9)

        xs = torch.split(x, self.img_concept_size, dim=-1)

        assert len(xs) == 3

        logits = []
        for i in range(3):
            logits.append(self.backbone(xs[i]))

        return torch.cat(logits, dim=-1), 0


class TripleMLP(nn.Module):
    NAME = "TripleMLP"

    def __init__(
        self,
        img_channels=3,
        hidden_channels=32,
        img_concept_size=28,
        latent_dim=8,
        label_dim=20,
        dropout=0.5,
    ):
        super(TripleMLP, self).__init__()

        self.img_concept_size = img_concept_size
        self.channels = 3
        self.img_channels = img_channels
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim
        self.label_dim = label_dim
        self.return_simple_concepts = False
        self.unflatten_dim = (3, 7)

        self.backbone = torch.nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=self.img_channels
                * self.img_concept_size
                * self.img_concept_size,
                out_features=256,
            ),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=self.latent_dim),
        )

    def forward(self, x):
        xs = torch.split(x, self.img_concept_size, dim=-1)

        assert len(xs) == 3, len(xs)

        logits = []

        if self.return_simple_concepts:
            for i in range(3):
                logits.append(self.backbone(xs[i]))
            logits = torch.cat(logits, dim=-1)
            logits = torch.split(logits, 6, dim=-1)
            logits = torch.stack(logits, dim=1)
            return logits

        for i in range(3):
            vars = torch.stack(torch.split(self.backbone(xs[i]), 3, dim=-1))
            logits.append(vars)

        logits = torch.stack(logits)

        for i in range(3):
            if i == 0:
                preds = logits[i]
            else:
                preds = torch.cat((preds, logits[i]), dim=-1)

        for i in range(2):
            if i == 0:
                c_preds = preds[i]
            else:
                c_preds = torch.cat((c_preds, preds[i]), dim=-1)

        return c_preds, 0