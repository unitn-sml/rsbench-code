import torch
import torch.nn as nn
import torch.nn.functional as F


class DisjointKANDCNN(nn.Module):
    def __init__(self, n_images):
        super(DisjointKANDCNN, self).__init__()
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
        self.fc1 = nn.Linear(128 * 1 * 5 * n_images, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.sigmoid = nn.Sigmoid()

        self.n_images = n_images

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(-1, 128 * 1 * 5)

        return x

    def _concatenate_embeddings(self, emb_list):
        assert len(emb_list) == self.n_images

        return torch.concatenate(emb_list, dim=1)

    def classify(self, emb_list):
        x = self._concatenate_embeddings(emb_list)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))

        complement = 1 - x
        x = torch.cat((x, complement), dim=1)

        return x


if __name__ == "__main__":
    model = DisjointKANDCNN(3)
    dummy_input_1 = torch.randn(64, 3, 28, 84)
    dummy_input_2 = torch.randn(64, 3, 28, 84)
    dummy_input_3 = torch.randn(64, 3, 28, 84)
    emb_1 = model(dummy_input_1)
    emb_2 = model(dummy_input_2)
    emb_3 = model(dummy_input_3)

    output = model.classify([emb_1, emb_2, emb_3])
    print(output.shape)
