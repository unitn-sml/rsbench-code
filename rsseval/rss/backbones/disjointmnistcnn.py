import torch
import torch.nn as nn
import torch.nn.functional as F


class DisjointMNISTAdditionCNN(nn.Module):
    def __init__(self, n_images):
        super(DisjointMNISTAdditionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.n_images = n_images
        self.fc1 = nn.Linear(32 * 7 * 7 * self.n_images, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 19)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 32 * 7 * 7)

        return x

    def _concatenate_embeddings(self, emb_list):
        assert len(emb_list) == self.n_images

        return torch.concatenate(emb_list, dim=1)

    def classify(self, emb_list):
        x = self._concatenate_embeddings(emb_list)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x


class MLP(nn.Module):
    def __init__(self, n_images):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 19)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)

        return x

    def _concatenate_embeddings(self, emb_list):
        assert len(emb_list) == self.n_images

        return torch.concatenate(emb_list, dim=1)


if __name__ == "__main__":
    model = DisjointMNISTAdditionCNN(n_images=2)
    dummy_input_1 = torch.randn(64, 1, 28, 28)
    dummy_input_2 = torch.randn(64, 1, 28, 28)
    emb_1 = model(dummy_input_1)
    emb_2 = model(dummy_input_2)
    output = model.classify([emb_1, emb_2])
    print(output.shape)
