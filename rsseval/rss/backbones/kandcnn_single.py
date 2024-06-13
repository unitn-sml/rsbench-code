import torch
import torch.nn as nn
import torch.nn.functional as F


class KANDCNNSingle(nn.Module):
    def __init__(self, n_images):
        super(KANDCNNSingle, self).__init__()
        self.n_images = n_images
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.conv5 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.fc1 = nn.Linear(n_images * 1024, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.sigmoid = nn.Sigmoid()

    def forward_embeddings(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        x = self.relu(self.conv5(x))
        x = self.pool(x)
        x = x.view(-1, 1024)
        return x

    def forward(self, x):
        embs = []

        # process each image separately
        for i in range(self.n_images):
            start = i * 64
            end = start + 64
            im = x[:, :, :, start:end]

            embs.append(self.forward_embeddings(im))

        embs = torch.concatenate(embs, dim=1)
        embs = self.relu(self.fc1(embs))
        embs = self.relu(self.fc2(embs))
        embs = self.sigmoid(self.fc3(embs))

        complement = 1 - embs
        embs = torch.cat((embs, complement), dim=1)

        return embs


if __name__ == "__main__":
    model = KANDCNNSingle(3)
    dummy_input = torch.randn(64, 3, 64, 192)
    output = model(dummy_input)
    print(output.shape)
