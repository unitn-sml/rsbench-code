import os
import torch
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np, glob
import re
import matplotlib.pyplot as plt
import joblib
from torchvision.datasets.folder import pil_loader


class MNMATHDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path,
        split,
        c_sup=1,
        which_c=[-1],
    ):
        
        self.base_path = base_path
        self.split = split

        # collecting images
        self.list_images = glob.glob(os.path.join(self.base_path, self.split, "*.png"))

        # sort the images
        self.list_images = sorted(self.list_images, key=self._extract_number)

        # ok transform
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.labels, self.concepts = [], []

        # lmao
        new_images = self.list_images.copy()

        # extract labels and concepts
        for item in self.list_images:
            name = os.path.splitext(os.path.basename(item))[0]
            # extract the ids out of the images
            meta_id = name.split("_")[-1]

            # get the target meta
            meta_scene = os.path.join(
                self.base_path,
                self.split,
                str(meta_id) + ".joblib",
            )

            if not os.path.exists(meta_scene):
                new_images.remove(
                    os.path.join(self.base_path, self.split, str(meta_id) + ".joblib")
                )
                continue

            # concepts and labels
            concepts, labels = [], []

            # load data from joblib
            data = joblib.load(meta_scene)

            # take the label
            label = data["label"]
            concept_values = data["meta"]["concepts"]

            converted_labels = [bool(l) for l in label]
            labels = np.array(converted_labels).astype(np.long)
            self.labels.append(labels)

            concepts = np.array(concept_values).astype(np.long)
            self.concepts.append(concepts)

        self.concepts = np.stack(self.concepts, axis=0)
        self.labels = np.stack(self.labels, axis=0)
        self.list_images = np.array(new_images)

    def _extract_number(self, path):
        match = re.search(r"\d+", path)
        return int(match.group()) if match else 0

    def __getitem__(self, item):

        labels = self.labels[item]
        concepts = self.concepts[item]
        img_path = self.list_images[item]
        image = pil_loader(img_path)

        # grayscale
        image = image.convert("L")

        return self.transform(image), labels, concepts

    def __len__(self):
        return len(self.list_images)


if __name__ == "__main__":
    print("Hello World")

    train_data = MNMATHDataset("../../data/xor_out_bits", "train")
    val_data = MNMATHDataset("../../data/xor_out_bits", "val")
    test_data = MNMATHDataset("../../data/xor_out_bits", "test")
    ood_data = MNMATHDataset("../../data/xor_out_bits", "ood")

    img, label, concepts = train_data[0]
    print(img.shape, concepts.shape, label.shape)

    plt.imshow(img.permute(1, 2, 0))
    plt.savefig("lmao.png")
    plt.close()
    quit()
