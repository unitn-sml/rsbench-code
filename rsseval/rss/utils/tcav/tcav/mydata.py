import os
from torch.utils.data import Dataset
from PIL import Image
import torch


class MyDataset(Dataset):
    """Dataset for concept image loading"""

    def __init__(self, base_path, transform=None, embedding=False):
        self.base_path = base_path
        self.transform = transform
        self.names_list = []
        for root, dirs, files in os.walk(base_path):
            self.names_list = files
        self.size = len(self.names_list)
        self.embedding = embedding

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image_path = os.path.join(self.base_path, self.names_list[idx])
        filename = self.names_list[idx]
        if not os.path.isfile(image_path):
            print(image_path + " does not exist!")
            return None
        if self.embedding:
            image = torch.load(image_path)
        else:
            with open(image_path, "rb") as f:
                with Image.open(f) as image:
                    image = image.convert("RGB")
            if self.transform:
                image = self.transform(image)
        return (image, filename)


class ValidateDataset(Dataset):
    def __init__(self, class_list, origin_dataloader, stype="default"):
        self.pixel = []
        self.label = []
        for data, label, concept in origin_dataloader:
            bs = data.size(0)
            for i in range(bs):
                x = data[i]
                l = label[i]

                if stype == "boia":
                    l = int("".join(map(str, label[i].to(torch.long).tolist())), 2)
                elif stype == "kand":
                    l = l[-1]

                if l in class_list:
                    self.pixel.append(x)
                    self.label.append(l)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.pixel[idx], self.label[idx]
