import torch
import torch.nn as nn
import torch.nn.functional as F


class CLEVRCnn(nn.Module):
    def __init__(self):
        super(CLEVRCnn, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 512, 3, padding=1)
        self.fc1 = nn.Linear(512 * 3 * 5, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv6(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 512 * 3 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))

        complement = 1 - x

        x = torch.stack(
            (
                x[:, 0],
                complement[:, 0],
            ),
            dim=1,
        )

        return x


if __name__ == "__main__":
    model = CLEVRCnn()
    dummy_input = torch.randn(64, 3, 240, 320)
    output = model(dummy_input)
    print(output.shape)
