import torch
import torch.nn as nn
import torch.nn.functional as F


class BOIAMLP(nn.Module):
    def __init__(self):
        super(BOIAMLP, self).__init__()
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, 4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.sigmoid(self.fc5(x))
        complement = 1 - x

        x = torch.stack(
            (
                x[:, 0],
                complement[:, 0],
                x[:, 1],
                complement[:, 1],
                x[:, 2],
                complement[:, 2],
                x[:, 3],
                complement[:, 3],
            ),
            dim=1,
        )

        return x


class CLIPMLP(nn.Module):
    def __init__(self):
        super(CLIPMLP, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        complement = 1 - x

        x = torch.stack(
            (
                x[:, 0],
                complement[:, 0],
                x[:, 1],
                complement[:, 1],
                x[:, 2],
                complement[:, 2],
                x[:, 3],
                complement[:, 3],
            ),
            dim=1,
        )

        return x


if __name__ == "__main__":
    model = BOIAMLP()
    dummy_input = torch.randn(64, 2048)
    output = model(dummy_input)
    print(output.shape)
